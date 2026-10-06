package proxy

import (
	"net/http"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/htang7415/MaxionBench/gateway/internal/config"
)

func TestTraceContextReachesUpstreamAndRouteIsRecorded(t *testing.T) {
	recorder := tracetest.NewSpanRecorder()
	tp := sdktrace.NewTracerProvider(sdktrace.WithSpanProcessor(recorder))
	prevTP, prevProp := otel.GetTracerProvider(), otel.GetTextMapPropagator()
	otel.SetTracerProvider(tp)
	otel.SetTextMapPropagator(propagation.TraceContext{})
	t.Cleanup(func() { otel.SetTracerProvider(prevTP); otel.SetTextMapPropagator(prevProp) })

	var mu sync.Mutex
	var upstreamParent string
	local := func(w http.ResponseWriter, r *http.Request) {
		mu.Lock()
		upstreamParent = r.Header.Get("traceparent")
		mu.Unlock()
		okLocal(w, r)
	}
	h := newHarness(t, local, config.LocalFirst, 4, 10)
	const traceID = "4bf92f3577b34da6a3ce929d0e0e4736"
	req, _ := http.NewRequest(http.MethodPost, h.srv.URL+"/v1/chat/completions",
		strings.NewReader(`{"messages":[{"role":"user","content":"hi"}],"max_tokens":4}`))
	req.Header.Set("traceparent", "00-"+traceID+"-00f067aa0ba902b7-01")
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	// The server span ends when the handler returns, which can be just after the client reads the body.
	for i := 0; i < 100 && len(recorder.Ended()) < 2; i++ {
		time.Sleep(10 * time.Millisecond)
	}

	mu.Lock()
	defer mu.Unlock()
	if !strings.HasPrefix(upstreamParent, "00-"+traceID+"-") {
		t.Fatalf("upstream traceparent %q does not continue trace %s", upstreamParent, traceID)
	}
	var server sdktrace.ReadOnlySpan
	for _, s := range recorder.Ended() {
		if s.SpanContext().TraceID().String() != traceID {
			t.Fatalf("span %q started a new trace", s.Name())
		}
		if s.SpanKind() == trace.SpanKindServer {
			server = s
		}
	}
	if server == nil {
		names := []string{}
		for _, s := range recorder.Ended() {
			names = append(names, s.Name())
		}
		t.Fatalf("no server span among %v", names)
	}
	attrs := map[string]string{}
	for _, kv := range server.Attributes() {
		attrs[string(kv.Key)] = kv.Value.Emit()
	}
	if attrs["maxion.backend"] != "local" || attrs["maxion.route_reason"] != "capacity" {
		t.Fatalf("route attributes missing: %v", attrs)
	}
	for k, v := range attrs {
		if strings.Contains(v, fakeKey) {
			t.Fatalf("attribute %s leaks the key", k)
		}
	}
}
