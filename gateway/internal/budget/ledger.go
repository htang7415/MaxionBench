// Package budget implements the paid-API spend ledger shared with the Python harness
// (maxionbench/harness/budget.py): an append-only JSONL file of reserve/commit/release events.
// Exposure = committed actual cost + open reservations. Every check-and-append holds an exclusive
// flock, so one hard cap holds across Go and Python processes.
package budget

import (
	"bufio"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sync"
	"syscall"
	"time"
)

// ErrBudgetExceeded is returned when a reservation would push exposure past the cap.
var ErrBudgetExceeded = errors.New("budget exceeded")

// Price is USD per 1M tokens.
type Price struct {
	InputPerM       float64 `yaml:"input_per_m"`
	OutputPerM      float64 `yaml:"output_per_m"`
	CachedInputPerM float64 `yaml:"cached_input_per_m"`
}

// Cost bills cached tokens (a subset of input tokens) at the cached rate.
func (p Price) Cost(inputTokens, outputTokens, cachedTokens int) float64 {
	if cachedTokens > inputTokens {
		cachedTokens = inputTokens
	}
	return (float64(inputTokens-cachedTokens)*p.InputPerM +
		float64(cachedTokens)*p.CachedInputPerM +
		float64(outputTokens)*p.OutputPerM) / 1e6
}

// Reservation is an open hold against the cap.
type Reservation struct {
	ID          string
	Label       string
	EstimateUSD float64
}

// Ledger is safe for concurrent use within a process and across processes.
type Ledger struct {
	path   string
	capUSD float64
	mu     sync.Mutex
}

// Open creates the ledger file (and parent directories) if needed.
func Open(path string, capUSD float64) (*Ledger, error) {
	if capUSD <= 0 {
		return nil, fmt.Errorf("cap must be > 0")
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return nil, err
	}
	f, err := os.OpenFile(path, os.O_CREATE|os.O_RDWR, 0o600)
	if err != nil {
		return nil, err
	}
	f.Close()
	return &Ledger{path: path, capUSD: capUSD}, nil
}

type event struct {
	At            string         `json:"at"`
	Event         string         `json:"event"`
	ReservationID string         `json:"reservation_id"`
	Label         string         `json:"label,omitempty"`
	EstimateUSD   *float64       `json:"estimate_usd,omitempty"`
	ActualUSD     *float64       `json:"actual_usd,omitempty"`
	Usage         map[string]any `json:"usage,omitempty"`
}

// withLock runs fn with the file open and flocked; exclusive for writers.
func (l *Ledger) withLock(exclusive bool, fn func(f *os.File) error) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	f, err := os.OpenFile(l.path, os.O_RDWR|os.O_APPEND, 0o600)
	if err != nil {
		return err
	}
	defer f.Close()
	how := syscall.LOCK_SH
	if exclusive {
		how = syscall.LOCK_EX
	}
	if err := syscall.Flock(int(f.Fd()), how); err != nil {
		return err
	}
	defer syscall.Flock(int(f.Fd()), syscall.LOCK_UN) //nolint:errcheck
	return fn(f)
}

func tally(f *os.File) (committed float64, open map[string]float64, err error) {
	open = map[string]float64{}
	if _, err = f.Seek(0, 0); err != nil {
		return 0, nil, err
	}
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<20), 1<<20)
	for sc.Scan() {
		if len(sc.Bytes()) == 0 {
			continue
		}
		var ev event
		if err := json.Unmarshal(sc.Bytes(), &ev); err != nil {
			return 0, nil, fmt.Errorf("corrupt ledger line: %w", err)
		}
		switch ev.Event {
		case "reserve":
			if ev.EstimateUSD != nil {
				open[ev.ReservationID] = *ev.EstimateUSD
			}
		case "commit":
			if ev.ActualUSD != nil {
				committed += *ev.ActualUSD
			}
			delete(open, ev.ReservationID)
		case "release":
			delete(open, ev.ReservationID)
		}
	}
	return committed, open, sc.Err()
}

func appendEvent(f *os.File, ev event) error {
	ev.At = time.Now().UTC().Format(time.RFC3339)
	b, err := json.Marshal(ev)
	if err != nil {
		return err
	}
	_, err = f.Write(append(b, '\n'))
	return err
}

func round6(v float64) float64 { return math.Round(v*1e6) / 1e6 }

// Remaining returns cap minus committed spend minus open reservations.
func (l *Ledger) Remaining() (float64, error) {
	var rem float64
	err := l.withLock(false, func(f *os.File) error {
		committed, open, err := tally(f)
		if err != nil {
			return err
		}
		rem = l.capUSD - committed
		for _, v := range open {
			rem -= v
		}
		return nil
	})
	return rem, err
}

// Reserve holds estimateUSD against the cap or returns ErrBudgetExceeded.
func (l *Ledger) Reserve(estimateUSD float64, label string) (Reservation, error) {
	var res Reservation
	err := l.withLock(true, func(f *os.File) error {
		committed, open, err := tally(f)
		if err != nil {
			return err
		}
		remaining := l.capUSD - committed
		for _, v := range open {
			remaining -= v
		}
		if estimateUSD > remaining {
			return fmt.Errorf("%w: estimate $%.6f > remaining $%.6f of $%.2f", ErrBudgetExceeded, estimateUSD, remaining, l.capUSD)
		}
		id := make([]byte, 16)
		if _, err := rand.Read(id); err != nil {
			return err
		}
		res = Reservation{ID: hex.EncodeToString(id), Label: label, EstimateUSD: estimateUSD}
		est := round6(estimateUSD)
		return appendEvent(f, event{Event: "reserve", ReservationID: res.ID, Label: label, EstimateUSD: &est})
	})
	return res, err
}

// Commit records the actual cost and closes the reservation.
func (l *Ledger) Commit(res Reservation, actualUSD float64, usage map[string]any) error {
	act := round6(actualUSD)
	return l.withLock(true, func(f *os.File) error {
		return appendEvent(f, event{Event: "commit", ReservationID: res.ID, Label: res.Label, ActualUSD: &act, Usage: usage})
	})
}

// Release closes a reservation that spent nothing.
func (l *Ledger) Release(res Reservation) error {
	return l.withLock(true, func(f *os.File) error {
		return appendEvent(f, event{Event: "release", ReservationID: res.ID})
	})
}
