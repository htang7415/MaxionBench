package proxy

import (
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
			StripFields: []string{"chat_template_kwargs", "ignore_eos", "cache_prompt"},
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
	body := `{"model":"qwen3","stream":true,"max_tokens":16,"ignore_eos":true,
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
	for _, f := range []string{"chat_template_kwargs", "ignore_eos"} {
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
