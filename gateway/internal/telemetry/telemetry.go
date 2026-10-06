// Package telemetry configures OpenTelemetry tracing for the gateway. W3C trace-context propagation
// is always on, so a client's traceparent reaches the fleet (llm-d, vLLM) and the remote unchanged;
// spans are exported over OTLP/HTTP only when OTEL_EXPORTER_OTLP_ENDPOINT is set.
package telemetry

import (
	"context"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracehttp"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.43.0"
)

// Setup installs the propagator and, if configured, an OTLP exporter. The returned function flushes
// and stops the exporter.
func Setup(ctx context.Context, service string) (func(context.Context) error, error) {
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(propagation.TraceContext{}, propagation.Baggage{}))
	if os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT") == "" && os.Getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT") == "" {
		return func(context.Context) error { return nil }, nil
	}
	exporter, err := otlptracehttp.New(ctx) // endpoint, headers and protocol from OTEL_* env vars
	if err != nil {
		return nil, err
	}
	res := resource.NewWithAttributes(semconv.SchemaURL, semconv.ServiceName(service))
	tp := sdktrace.NewTracerProvider(sdktrace.WithBatcher(exporter), sdktrace.WithResource(res))
	otel.SetTracerProvider(tp)
	return tp.Shutdown, nil
}
