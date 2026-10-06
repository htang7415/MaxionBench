package proxy

import (
	"sync"
	"time"
)

// serviceEstimator tracks local completions over a sliding window. Its per-request service time is
// the fleet's recent time per completed request (window / completions, i.e. 1 / throughput), so
// in-flight x service time is a Little's-law estimate of how long a new request waits locally. It
// needs no knowledge of how many slots the fleet has.
type serviceEstimator struct {
	mu         sync.Mutex
	window     time.Duration
	minSamples int
	started    time.Time
	done       []time.Time // completion times, oldest first
}

func newServiceEstimator(window time.Duration, minSamples int, now time.Time) *serviceEstimator {
	return &serviceEstimator{window: window, minSamples: minSamples, started: now}
}

func (e *serviceEstimator) record(now time.Time) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.done = append(e.done, now)
	e.trim(now)
}

// perRequest returns the recent seconds of fleet time per completed request, or ok=false while there
// are fewer than minSamples completions in the window.
func (e *serviceEstimator) perRequest(now time.Time) (float64, bool) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.trim(now)
	if len(e.done) < e.minSamples {
		return 0, false
	}
	span := min(e.window, now.Sub(e.started)) // shorter right after start-up
	return span.Seconds() / float64(len(e.done)), true
}

func (e *serviceEstimator) trim(now time.Time) {
	cut := 0
	for cut < len(e.done) && now.Sub(e.done[cut]) >= e.window { // window is (now-window, now]
		cut++
	}
	e.done = e.done[cut:]
}
