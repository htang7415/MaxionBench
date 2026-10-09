package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"math"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/htang7415/MaxionBench/gateway/internal/budget"
	"github.com/htang7415/MaxionBench/gateway/internal/config"
	"github.com/htang7415/MaxionBench/gateway/internal/ctxmgr"
)

const fakeKey = "FAKE-key-0123456789abcdefghijklmnop"

func sse(w http.ResponseWriter, text string, usage map[string]any) {
	w.Header().Set("Content-Type", "text/event-stream")
	w.WriteHeader(http.StatusOK)
	fmt.Fprintf(w, "data: {\"choices\":[{\"delta\":{\"content\":%q}}]}\n\n", text)
	if usage != nil {
		b, _ := json.Marshal(map[string]any{"choices": []any{}, "usage": usage})
		fmt.Fprintf(w, "data: %s\n\n", b)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
}

type fakeRemote struct {
	mu      sync.Mutex
	bodies  []map[string]any
	auth    []string
	status  int
	errBody string
}

func (f *fakeRemote) handler(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
	f.mu.Lock()
	f.bodies = append(f.bodies, body)
	f.auth = append(f.auth, r.Header.Get("Authorization"))
	status, errBody := f.status, f.errBody
	f.mu.Unlock()
	if status != 0 && status != http.StatusOK {
		w.WriteHeader(status)
		io.WriteString(w, errBody) //nolint:errcheck
		return
	}
	sse(w, "remote", map[string]any{"prompt_tokens": 1000, "completion_tokens": 10,
		"prompt_tokens_details": map[string]any{"cached_tokens": 200}})
}

type harness struct {
	gw     *Gateway
	srv    *httptest.Server
	remote *fakeRemote
	ledger *budget.Ledger
	reg    *prometheus.Registry
}

func newHarness(t *testing.T, local http.HandlerFunc, policy string, maxInflight int, capUSD float64) *harness {
	t.Helper()
	localSrv := httptest.NewServer(local)
	t.Cleanup(localSrv.Close)
	fr := &fakeRemote{}
	remoteSrv := httptest.NewServer(http.HandlerFunc(fr.handler))
	t.Cleanup(remoteSrv.Close)
	cfg := &config.Config{
		Policy: policy, Failover: true,
		Local: config.Local{Upstreams: []string{localSrv.URL}, Model: "qwen3-0.6b", MaxInflight: maxInflight, TimeoutS: 10},
		Remote: config.Remote{Enabled: true, BaseURL: remoteSrv.URL, ChatPath: "/chat/completions",
			Model: "gemini-3.5-flash-lite", ReasoningEffort: "minimal", TimeoutS: 10,
			StripFields: []string{"chat_template_kwargs", "ignore_eos", "cache_prompt", "prompt_cache_key"},
			Price:       budget.Price{InputPerM: 1.0, OutputPerM: 10.0, CachedInputPerM: 0.1}, CapUSD: capUSD},
	}
	ledger, err := budget.Open(filepath.Join(t.TempDir(), "ledger.jsonl"), capUSD)
	if err != nil {
		t.Fatal(err)
	}
	reg := prometheus.NewRegistry()
	gw := New(cfg, fakeKey, ledger, reg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	srv := httptest.NewServer(gw.Handler(reg))
	t.Cleanup(srv.Close)
	return &harness{gw: gw, srv: srv, remote: fr, ledger: ledger, reg: reg}
}

func (h *harness) post(t *testing.T) (*http.Response, string) {
	t.Helper()
	body := `{"model":"qwen3","stream":true,"max_tokens":16,"ignore_eos":true,"prompt_cache_key":"s1",
	"chat_template_kwargs":{"enable_thinking":false},"messages":[{"role":"user","content":"` + strings.Repeat("x", 300) + `"}]}`
	resp, err := http.Post(h.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	b, _ := io.ReadAll(resp.Body)
	resp.Body.Close()
	return resp, string(b)
}

func okLocal(w http.ResponseWriter, r *http.Request) {
	var body map[string]any
	json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
	sse(w, "local:"+fmt.Sprint(body["model"]), nil)
}

func TestLocalFirstServesLocallyUnderCapacity(t *testing.T) {
	h := newHarness(t, okLocal, config.LocalFirst, 4, 10)
	resp, body := h.post(t)
	if resp.Header.Get(BackendHeader) != "local" || resp.Header.Get(ReasonHeader) != "capacity" {
		t.Fatalf("headers: %v", resp.Header)
	}
	if !strings.Contains(body, "local:qwen3-0.6b") { // model rewritten for the fleet
		t.Fatalf("body: %s", body)
	}
	if len(h.remote.bodies) != 0 {
		t.Fatal("remote must not be called under capacity")
	}
}

func TestOverflowToRemoteWhenSaturatedAndBillActualUsage(t *testing.T) {
	release := make(chan struct{})
	slowLocal := func(w http.ResponseWriter, r *http.Request) { <-release; okLocal(w, r) }
	h := newHarness(t, slowLocal, config.LocalFirst, 1, 10)
	done := make(chan struct{})
	go func() { h.post(t); close(done) }() // occupies the only local slot
	time.Sleep(100 * time.Millisecond)

	resp, body := h.post(t)
	close(release)
	<-done
	if resp.Header.Get(BackendHeader) != "remote" || resp.Header.Get(ReasonHeader) != "local_saturated" {
		t.Fatalf("headers: %v", resp.Header)
	}
	if !strings.Contains(body, "remote") {
		t.Fatalf("body: %s", body)
	}
	sent := h.remote.bodies[0]
	if sent["model"] != "gemini-3.5-flash-lite" || sent["reasoning_effort"] != "minimal" {
		t.Fatalf("remote body: %v", sent)
	}
	for _, f := range []string{"chat_template_kwargs", "ignore_eos", "prompt_cache_key"} {
		if _, ok := sent[f]; ok {
			t.Fatalf("engine field %q leaked to provider", f)
		}
	}
	if so, _ := sent["stream_options"].(map[string]any); so["include_usage"] != true {
		t.Fatalf("include_usage not forced: %v", sent["stream_options"])
	}
	if h.remote.auth[0] != "Bearer "+fakeKey {
		t.Fatal("remote did not receive bearer key")
	}
	want := (800*1.0 + 200*0.1 + 10*10.0) / 1e6
	if rem, _ := h.ledger.Remaining(); math.Abs((10-rem)-want) > 1e-9 {
		t.Fatalf("spent %v, want %v", 10-rem, want)
	}
}

func TestFailoverToRemoteWhenLocalErrors(t *testing.T) {
	broken := func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusInternalServerError) }
	h := newHarness(t, broken, config.LocalFirst, 4, 10)
	resp, _ := h.post(t)
	if resp.StatusCode != 200 || resp.Header.Get(ReasonHeader) != "local_failover" {
		t.Fatalf("status %d headers %v", resp.StatusCode, resp.Header)
	}
}

func TestBudgetExhaustedQueuesLocallyAndRemoteOnlyReturns429(t *testing.T) {
	release := make(chan struct{})
	slowLocal := func(w http.ResponseWriter, r *http.Request) { <-release; okLocal(w, r) }
	h := newHarness(t, slowLocal, config.LocalFirst, 1, 1e-9) // cap below any request's estimate
	done := make(chan struct{})
	go func() { h.post(t); close(done) }()
	time.Sleep(100 * time.Millisecond)
	go func() { time.Sleep(200 * time.Millisecond); close(release) }()
	resp, _ := h.post(t)
	<-done
	if resp.Header.Get(BackendHeader) != "local" || resp.Header.Get(ReasonHeader) != "budget_exhausted" {
		t.Fatalf("headers: %v", resp.Header)
	}
	if len(h.remote.bodies) != 0 {
		t.Fatal("remote called despite exhausted budget")
	}

	ro := newHarness(t, okLocal, config.RemoteOnly, 1, 1e-9)
	if resp, _ := ro.post(t); resp.StatusCode != http.StatusTooManyRequests {
		t.Fatalf("remote_only with no budget: status %d", resp.StatusCode)
	}
}

func TestProviderErrorIsRedactedAndNotBilled(t *testing.T) {
	h := newHarness(t, okLocal, config.RemoteOnly, 1, 10)
	h.remote.status, h.remote.errBody = 401, `{"error":"bad key `+fakeKey+`"}`
	resp, body := h.post(t)
	if resp.StatusCode != 401 || strings.Contains(body, fakeKey) || !strings.Contains(body, redacted) {
		t.Fatalf("status %d body %s", resp.StatusCode, body)
	}
	if rem, _ := h.ledger.Remaining(); rem != 10 {
		t.Fatalf("provider error was billed: remaining %v", rem)
	}
	m, _ := http.Get(h.srv.URL + "/metrics")
	mb, _ := io.ReadAll(m.Body)
	if strings.Contains(string(mb), fakeKey) || !strings.Contains(string(mb), "maxion_gateway_route_decisions_total") {
		t.Fatal("metrics leaked key or missing route counters")
	}
}

func TestSLOPolicyOverflowsOnPredictedWait(t *testing.T) {
	h := newHarness(t, okLocal, config.LocalFirstSLO, 100, 10)
	h.gw.cfg.Local.SLOTTFTS = 1.0
	now := time.Now()
	h.gw.est = newServiceEstimator(10*time.Second, 4, now.Add(-time.Minute))
	if r := h.gw.overflowReason(context.Background(), now); r != "" {
		t.Fatalf("no samples yet: got %q", r)
	}
	for i := 0; i < 20; i++ { // 20 completions in 10 s -> 0.5 s per request
		h.gw.est.record(now.Add(-time.Duration(i) * 400 * time.Millisecond))
	}
	h.gw.local.Store(2) // 2 x 0.5 = 1.0 s: at the SLO, stay local
	if r := h.gw.overflowReason(context.Background(), now); r != "" {
		t.Fatalf("predicted 1.0 s: got %q", r)
	}
	h.gw.local.Store(3) // 1.5 s > 1.0 s
	if r := h.gw.overflowReason(context.Background(), now); r != "slo_predicted" {
		t.Fatalf("predicted 1.5 s: got %q", r)
	}
	h.gw.local.Store(100) // the in-flight cap still applies
	if r := h.gw.overflowReason(context.Background(), now); r != "local_saturated" {
		t.Fatalf("at cap: got %q", r)
	}
	h.gw.local.Store(0)

	lf := newHarness(t, okLocal, config.LocalFirst, 100, 10) // fixed threshold ignores the prediction
	lf.gw.est = h.gw.est
	lf.gw.local.Store(50)
	if r := lf.gw.overflowReason(context.Background(), now); r != "" {
		t.Fatalf("local_first below cap: got %q", r)
	}
}

func TestSLOPolicyRoutesRemoteEndToEnd(t *testing.T) {
	release := make(chan struct{})
	slowLocal := func(w http.ResponseWriter, r *http.Request) { <-release; okLocal(w, r) }
	h := newHarness(t, slowLocal, config.LocalFirstSLO, 100, 10)
	h.gw.cfg.Local.SLOTTFTS = 0.1
	h.gw.est = newServiceEstimator(10*time.Second, 1, time.Now().Add(-10*time.Second))
	h.gw.est.record(time.Now()) // 1 completion in 10 s -> 10 s per request
	done := make(chan struct{})
	go func() { h.post(t); close(done) }() // one in flight -> predicted wait 10 s
	time.Sleep(100 * time.Millisecond)
	resp, _ := h.post(t)
	close(release)
	<-done
	if resp.Header.Get(BackendHeader) != "remote" || resp.Header.Get(ReasonHeader) != "slo_predicted" {
		t.Fatalf("headers: %v", resp.Header)
	}
	m, _ := http.Get(h.srv.URL + "/metrics")
	mb, _ := io.ReadAll(m.Body)
	if !strings.Contains(string(mb), "maxion_gateway_predicted_local_wait_seconds 10") {
		t.Fatal("predicted wait gauge missing")
	}
}

func TestUsageBillsHiddenReasoningTokens(t *testing.T) {
	var u usage
	parseUsageJSON([]byte(`{"usage":{"prompt_tokens":31,"completion_tokens":3,"total_tokens":232}}`), &u)
	if u.Reasoning != 198 || u.Completion != 3 {
		t.Fatalf("Gemini-style totals: %+v", u)
	}
	var v usage
	parseUsageJSON([]byte(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,`+
		`"completion_tokens_details":{"reasoning_tokens":4}}}`), &v)
	if v.Reasoning != 4 {
		t.Fatalf("explicit reasoning count: %+v", v)
	}
	var w usage
	parseUsageJSON([]byte(`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}}`), &w)
	if w.Reasoning != 0 {
		t.Fatalf("consistent totals must add nothing: %+v", w)
	}
}

func TestContextManagerRewritesAgentHistory(t *testing.T) {
	var mu sync.Mutex
	var seen [][]any
	capture := func(w http.ResponseWriter, r *http.Request) {
		var body map[string]any
		json.NewDecoder(r.Body).Decode(&body) //nolint:errcheck
		mu.Lock()
		seen = append(seen, body["messages"].([]any))
		mu.Unlock()
		sse(w, "ok", nil)
	}
	h := newHarness(t, capture, config.LocalOnly, 4, 10)
	if resp, _ := h.post(t); resp.Header.Get(ContextHeader) != "" {
		t.Fatal("context header set while context management is off")
	}
	h.gw.ctx = ctxmgr.New(ctxmgr.Config{Policy: "mask+cache", Keep: 1, BudgetTokens: 64_000, MaxSessions: 10})

	history := []any{map[string]any{"role": "system", "content": "sys"}, map[string]any{"role": "user", "content": "task"}}
	send := func() string {
		b, _ := json.Marshal(map[string]any{"model": "m", "prompt_cache_key": "agent-1", "messages": history})
		resp, err := http.Post(h.srv.URL+"/v1/chat/completions", "application/json", strings.NewReader(string(b)))
		if err != nil {
			t.Fatal(err)
		}
		io.Copy(io.Discard, resp.Body) //nolint:errcheck
		resp.Body.Close()
		return resp.Header.Get(ContextHeader)
	}
	var actions []string
	for e := range 3 {
		actions = append(actions, send())
		id := fmt.Sprint(e)
		history = append(history,
			map[string]any{"role": "assistant", "content": nil, "tool_calls": []any{map[string]any{"id": id, "type": "function",
				"function": map[string]any{"name": "read", "arguments": "{}"}}}},
			map[string]any{"role": "tool", "tool_call_id": id, "content": strings.Repeat("z", 1000)})
	}
	if strings.Join(actions, ",") != "start,append,append" {
		t.Fatalf("actions %v", actions)
	}
	if got := seen[len(seen)-1]; len(got) != 6 || got[3].(map[string]any)["content"] != strings.Repeat("z", 1000) {
		t.Fatal("under budget the upstream must get the client's history unchanged")
	}

	h.gw.ctx = ctxmgr.New(ctxmgr.Config{Policy: "mask+cache", Keep: 1, BudgetTokens: 400, MaxSessions: 10})
	if a := send(); a != ctxmgr.Start {
		t.Fatalf("action %s", a)
	}
	got := seen[len(seen)-1]
	if got[3].(map[string]any)["content"] != ctxmgr.MaskText || got[7].(map[string]any)["content"] == ctxmgr.MaskText {
		t.Fatal("over budget: old tool results masked, the newest kept")
	}
}
