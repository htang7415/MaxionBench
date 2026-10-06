package proxy

import (
	"math"
	"testing"
	"time"
)

func TestServiceEstimatorUsesWindowedThroughput(t *testing.T) {
	t0 := time.Unix(1000, 0)
	e := newServiceEstimator(10*time.Second, 3, t0)
	if _, ok := e.perRequest(t0.Add(time.Second)); ok {
		t.Fatal("no prediction expected before min samples")
	}
	for i := 1; i <= 20; i++ { // one completion per second for 20 s; only the last 10 s count
		e.record(t0.Add(time.Duration(i) * time.Second))
	}
	got, ok := e.perRequest(t0.Add(20 * time.Second))
	if !ok || math.Abs(got-1.0) > 1e-9 { // 10 completions in the 10 s window -> 1 s per request
		t.Fatalf("perRequest = %v, %v", got, ok)
	}
	if _, ok := e.perRequest(t0.Add(40 * time.Second)); ok {
		t.Fatal("old completions must age out of the window")
	}
}

func TestServiceEstimatorShortSpanAfterStart(t *testing.T) {
	t0 := time.Unix(0, 0)
	e := newServiceEstimator(10*time.Second, 2, t0)
	for i := 0; i < 4; i++ {
		e.record(t0.Add(500 * time.Millisecond))
	}
	if got, _ := e.perRequest(t0.Add(2 * time.Second)); math.Abs(got-0.5) > 1e-9 { // 2 s up, 4 done
		t.Fatalf("perRequest = %v", got)
	}
}
