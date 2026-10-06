// Package proxy is an OpenAI-compatible AI gateway: local-first routing to a self-hosted fleet
// (vLLM replicas or an llm-d gateway) with SLO-driven overflow and failover to a paid remote API,
// under a hard spend cap shared with the Python harness.
package proxy

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"sync/atomic"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promhttp"
	"go.opentelemetry.io/contrib/instrumentation/net/http/otelhttp"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/htang7415/MaxionBench/gateway/internal/budget"
	"github.com/htang7415/MaxionBench/gateway/internal/config"
)

const (
	BackendHeader  = "X-Maxionbench-Backend"
	ReasonHeader   = "X-Maxionbench-Route-Reason"
	maxBodyBytes   = 8 << 20
	defaultMaxToks = 256
	redacted       = "[REDACTED]"
)

var errBudget = errors.New("remote budget exhausted")

type metrics struct {
	requests  *prometheus.CounterVec
	routes    *prometheus.CounterVec
	inflight  *prometheus.GaugeVec
	ttfb      *prometheus.HistogramVec
	spend     prometheus.Counter
	remaining prometheus.Gauge
	predicted prometheus.Gauge
}

func newMetrics(reg prometheus.Registerer) *metrics {
	m := &metrics{
		requests: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "maxion_gateway_requests_total", Help: "Requests by backend and HTTP status."}, []string{"backend", "code"}),
		routes: prometheus.NewCounterVec(prometheus.CounterOpts{
			Name: "maxion_gateway_route_decisions_total", Help: "Routing decisions by backend and reason."}, []string{"backend", "reason"}),
		inflight: prometheus.NewGaugeVec(prometheus.GaugeOpts{
			Name: "maxion_gateway_inflight", Help: "In-flight requests by backend."}, []string{"backend"}),
		ttfb: prometheus.NewHistogramVec(prometheus.HistogramOpts{
			Name: "maxion_gateway_upstream_ttfb_seconds", Help: "Time to first upstream body byte.",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 14)}, []string{"backend"}),
		spend: prometheus.NewCounter(prometheus.CounterOpts{
			Name: "maxion_gateway_remote_spend_usd_total", Help: "Committed remote spend in USD."}),
		remaining: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "maxion_gateway_budget_remaining_usd", Help: "Remaining remote budget (cap - spend - holds)."}),
		predicted: prometheus.NewGauge(prometheus.GaugeOpts{
			Name: "maxion_gateway_predicted_local_wait_seconds", Help: "Last predicted local wait (local_first_slo)."}),
	}
	reg.MustRegister(m.requests, m.routes, m.inflight, m.ttfb, m.spend, m.remaining, m.predicted)
	return m
}

// Gateway handles chat completions.
type Gateway struct {
	cfg      *config.Config
	key      string // never logged; redacted from forwarded error bodies
	ledger   *budget.Ledger
	client   *http.Client
	m        *metrics
	local    atomic.Int64
	upstream []atomic.Int64
	est      *serviceEstimator
	log      *slog.Logger
}

// New builds a gateway. key and ledger may be empty/nil when the remote is disabled.
func New(cfg *config.Config, key string, ledger *budget.Ledger, reg *prometheus.Registry, log *slog.Logger) *Gateway {
	g := &Gateway{
		cfg: cfg, key: key, ledger: ledger, log: log,
		// The instrumented transport opens a client span per upstream call and injects traceparent.
		client:   &http.Client{Transport: otelhttp.NewTransport(&http.Transport{MaxIdleConnsPerHost: 64})},
		m:        newMetrics(reg),
		upstream: make([]atomic.Int64, len(cfg.Local.Upstreams)),
		est:      newServiceEstimator(seconds(cfg.Local.WindowS), cfg.Local.MinSamples, time.Now()),
	}
	g.refreshRemaining()
	return g
}

// Handler serves /v1/chat/completions, /health and /metrics.
func (g *Gateway) Handler(reg *prometheus.Registry) http.Handler {
	mux := http.NewServeMux()
	mux.Handle("POST /v1/chat/completions", otelhttp.NewHandler(http.HandlerFunc(g.chat), "chat.completions"))
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.Handle("GET /metrics", promhttp.HandlerFor(reg, promhttp.HandlerOpts{}))
	return mux
}

func (g *Gateway) chat(w http.ResponseWriter, r *http.Request) {
	raw, err := io.ReadAll(io.LimitReader(r.Body, maxBodyBytes))
	if err != nil {
		writeError(w, http.StatusBadRequest, "unreadable body")
		return
	}
	var body map[string]any
	if err := json.Unmarshal(raw, &body); err != nil {
		writeError(w, http.StatusBadRequest, "body must be a JSON object")
		return
	}
	remoteOK := g.cfg.Remote.Enabled
	switch g.cfg.Policy {
	case config.RemoteOnly:
		if err := g.serveRemote(r.Context(), w, body, "policy"); errors.Is(err, errBudget) {
			writeError(w, http.StatusTooManyRequests, "remote budget exhausted")
		}
	case config.LocalOnly:
		g.serveLocal(r.Context(), w, body, "policy", false)
	default: // local_first, local_first_slo
		reason := g.overflowReason(r.Context(), time.Now())
		if reason == "" || !remoteOK {
			g.serveLocal(r.Context(), w, body, "capacity", remoteOK && g.cfg.Failover)
			return
		}
		if err := g.serveRemote(r.Context(), w, body, reason); errors.Is(err, errBudget) {
			g.serveLocal(r.Context(), w, body, "budget_exhausted", false) // queue locally instead of overspending
		}
	}
}

// overflowReason returns why a new request should go remote, or "" to serve it locally.
func (g *Gateway) overflowReason(ctx context.Context, now time.Time) string {
	inflight := g.local.Load()
	if inflight >= int64(g.cfg.Local.MaxInflight) {
		return "local_saturated"
	}
	if g.cfg.Policy != config.LocalFirstSLO {
		return ""
	}
	perReq, ok := g.est.perRequest(now)
	if !ok {
		return "" // too few recent completions to predict; the in-flight cap still applies
	}
	wait := float64(inflight) * perReq
	g.m.predicted.Set(wait)
	trace.SpanFromContext(ctx).SetAttributes(attribute.Float64("maxion.predicted_local_wait_s", wait))
	if wait > g.cfg.Local.SLOTTFTS {
		return "slo_predicted"
	}
	return ""
}

func (g *Gateway) pickUpstream() int {
	best := 0
	for i := range g.upstream {
		if g.upstream[i].Load() < g.upstream[best].Load() {
			best = i
		}
	}
	return best
}

// serveLocal proxies to the fleet. With failover, a connection error or 5xx that occurs before any
// byte reaches the client is retried on the remote.
func (g *Gateway) serveLocal(ctx context.Context, w http.ResponseWriter, body map[string]any, reason string, failover bool) {
	idx := g.pickUpstream()
	g.local.Add(1)
	g.upstream[idx].Add(1)
	g.m.inflight.WithLabelValues("local").Inc()
	defer func() {
		g.local.Add(-1)
		g.upstream[idx].Add(-1)
		g.m.inflight.WithLabelValues("local").Dec()
	}()

	out := cloneMap(body)
	if g.cfg.Local.Model != "" {
		out["model"] = g.cfg.Local.Model
	}
	payload, _ := json.Marshal(out)
	ctx, cancel := context.WithTimeout(ctx, seconds(g.cfg.Local.TimeoutS))
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(g.cfg.Local.Upstreams[idx], "/")+"/v1/chat/completions", bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	start := time.Now()
	resp, err := g.client.Do(req)
	if err != nil || resp.StatusCode >= 500 {
		status := "connect_error"
		if err == nil {
			status = fmt.Sprintf("http_%d", resp.StatusCode)
			io.Copy(io.Discard, resp.Body) //nolint:errcheck
			resp.Body.Close()
		}
		g.log.Warn("local upstream failed", "upstream", idx, "status", status)
		if failover {
			if ferr := g.serveRemote(ctx, w, body, "local_failover"); ferr == nil || !errors.Is(ferr, errBudget) {
				return
			}
		}
		g.m.requests.WithLabelValues("local", "502").Inc()
		writeError(w, http.StatusBadGateway, "local upstream unavailable")
		return
	}
	defer resp.Body.Close()
	g.m.routes.WithLabelValues("local", reason).Inc()
	routeSpan(ctx, "local", reason)
	copyHeaders(w, resp, "local", reason)
	w.WriteHeader(resp.StatusCode)
	g.m.requests.WithLabelValues("local", fmt.Sprint(resp.StatusCode)).Inc()
	streamCopy(w, resp.Body, func() { g.m.ttfb.WithLabelValues("local").Observe(time.Since(start).Seconds()) }, nil)
	if resp.StatusCode < 400 {
		g.est.record(time.Now())
	}
}

// serveRemote reserves the worst-case cost, proxies to the paid provider, and commits actual usage.
// It returns errBudget (having written nothing) if the reservation is refused.
func (g *Gateway) serveRemote(ctx context.Context, w http.ResponseWriter, body map[string]any, reason string) error {
	rc := g.cfg.Remote
	maxTok := intField(body, "max_tokens", defaultMaxToks)
	estIn := estimateTokens(body)
	estimate := rc.Price.Cost(estIn, maxTok, 0)
	res, err := g.ledger.Reserve(estimate, "gateway/"+reason)
	if err != nil {
		g.m.routes.WithLabelValues("remote", "refused_"+reason).Inc()
		g.log.Warn("remote reservation refused", "reason", reason, "err", err.Error())
		return errBudget
	}
	g.m.inflight.WithLabelValues("remote").Inc()
	defer g.m.inflight.WithLabelValues("remote").Dec()
	defer g.refreshRemaining()

	out := cloneMap(body)
	out["model"] = rc.Model
	for _, f := range rc.StripFields {
		delete(out, f)
	}
	if rc.ReasoningEffort != "" {
		out["reasoning_effort"] = rc.ReasoningEffort
	}
	for k, v := range rc.Extra {
		out[k] = v
	}
	stream, _ := out["stream"].(bool)
	if stream {
		out["stream_options"] = map[string]any{"include_usage": true}
	}
	payload, _ := json.Marshal(out)
	ctx, cancel := context.WithTimeout(ctx, seconds(rc.TimeoutS))
	defer cancel()
	req, _ := http.NewRequestWithContext(ctx, http.MethodPost,
		strings.TrimRight(rc.BaseURL, "/")+rc.ChatPath, bytes.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+g.key)
	start := time.Now()
	resp, err := g.client.Do(req)
	if err != nil {
		g.ledger.Release(res) //nolint:errcheck // nothing was billed
		g.m.requests.WithLabelValues("remote", "502").Inc()
		writeError(w, http.StatusBadGateway, g.redact("remote unavailable: "+err.Error()))
		return nil
	}
	defer resp.Body.Close()
	g.m.routes.WithLabelValues("remote", reason).Inc()
	routeSpan(ctx, "remote", reason)
	if resp.StatusCode != http.StatusOK {
		msg, _ := io.ReadAll(io.LimitReader(resp.Body, 64<<10))
		g.ledger.Release(res) //nolint:errcheck // provider errors are not billed
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set(BackendHeader, "remote")
		w.Header().Set(ReasonHeader, reason)
		w.WriteHeader(resp.StatusCode)
		w.Write([]byte(g.redact(string(msg)))) //nolint:errcheck
		g.m.requests.WithLabelValues("remote", fmt.Sprint(resp.StatusCode)).Inc()
		return nil
	}
	copyHeaders(w, resp, "remote", reason)
	w.WriteHeader(http.StatusOK)
	g.m.requests.WithLabelValues("remote", "200").Inc()
	var u usage
	streamCopy(w, resp.Body, func() { g.m.ttfb.WithLabelValues("remote").Observe(time.Since(start).Seconds()) }, &u)

	cost, meta := estimate, map[string]any{"estimated": true, "reason": reason}
	if u.found {
		cost = rc.Price.Cost(u.Prompt, u.Completion, u.Cached)
		meta = map[string]any{"input_tokens": u.Prompt, "cached_tokens": u.Cached, "output_tokens": u.Completion, "reason": reason}
	}
	if err := g.ledger.Commit(res, cost, meta); err != nil {
		g.log.Error("ledger commit failed", "err", err.Error())
	}
	g.m.spend.Add(cost)
	return nil
}

// routeSpan tags the request's server span with the routing decision (never headers or bodies).
func routeSpan(ctx context.Context, backend, reason string) {
	trace.SpanFromContext(ctx).SetAttributes(attribute.String("maxion.backend", backend),
		attribute.String("maxion.route_reason", reason))
}

func (g *Gateway) refreshRemaining() {
	if g.ledger == nil {
		return
	}
	if rem, err := g.ledger.Remaining(); err == nil {
		g.m.remaining.Set(rem)
	}
}

func (g *Gateway) redact(s string) string {
	if g.key == "" {
		return s
	}
	return strings.ReplaceAll(s, g.key, redacted)
}

type usage struct {
	Prompt, Completion, Cached int
	found                      bool
}

// streamCopy forwards the upstream body chunk by chunk, flushing each one; with u != nil it also
// parses SSE (or a plain JSON body) to capture the provider-reported usage.
func streamCopy(w http.ResponseWriter, src io.Reader, firstByte func(), u *usage) {
	flusher, _ := w.(http.Flusher)
	br := bufio.NewReaderSize(src, 32<<10)
	first := true
	var whole bytes.Buffer
	for {
		line, err := br.ReadBytes('\n')
		if len(line) > 0 {
			if first {
				firstByte()
				first = false
			}
			w.Write(line) //nolint:errcheck
			if flusher != nil {
				flusher.Flush()
			}
			if u != nil {
				whole.Write(line)
				parseUsageLine(line, u)
			}
		}
		if err != nil {
			break
		}
	}
	if u != nil && !u.found { // non-streaming JSON response
		parseUsageJSON(whole.Bytes(), u)
	}
}

func parseUsageLine(line []byte, u *usage) {
	s := bytes.TrimSpace(line)
	if !bytes.HasPrefix(s, []byte("data:")) {
		return
	}
	parseUsageJSON(bytes.TrimSpace(s[len("data:"):]), u)
}

func parseUsageJSON(b []byte, u *usage) {
	var ev struct {
		Usage *struct {
			PromptTokens        int `json:"prompt_tokens"`
			CompletionTokens    int `json:"completion_tokens"`
			PromptTokensDetails *struct {
				CachedTokens int `json:"cached_tokens"`
			} `json:"prompt_tokens_details"`
		} `json:"usage"`
	}
	if json.Unmarshal(b, &ev) != nil || ev.Usage == nil || ev.Usage.PromptTokens == 0 {
		return
	}
	u.Prompt, u.Completion, u.found = ev.Usage.PromptTokens, ev.Usage.CompletionTokens, true
	if ev.Usage.PromptTokensDetails != nil {
		u.Cached = ev.Usage.PromptTokensDetails.CachedTokens
	}
}

// estimateTokens is a conservative prompt estimate (~3 chars/token + per-message overhead), matching
// the Python harness's estimator.
func estimateTokens(body map[string]any) int {
	msgs, _ := body["messages"].([]any)
	total := 0
	for _, m := range msgs {
		mm, _ := m.(map[string]any)
		content, _ := mm["content"].(string)
		total += len(content)/3 + 8
	}
	return total
}

func intField(body map[string]any, key string, def int) int {
	if v, ok := body[key].(float64); ok && v > 0 {
		return int(v)
	}
	return def
}

func cloneMap(m map[string]any) map[string]any {
	out := make(map[string]any, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func copyHeaders(w http.ResponseWriter, resp *http.Response, backend, reason string) {
	if ct := resp.Header.Get("Content-Type"); ct != "" {
		w.Header().Set("Content-Type", ct)
	}
	w.Header().Set(BackendHeader, backend)
	w.Header().Set(ReasonHeader, reason)
}

func writeError(w http.ResponseWriter, code int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	json.NewEncoder(w).Encode(map[string]any{"error": map[string]any{"message": msg, "code": code}}) //nolint:errcheck
}

func seconds(s float64) time.Duration { return time.Duration(s * float64(time.Second)) }
