package budget

import (
	"errors"
	"math"
	"os"
	"path/filepath"
	"sync"
	"testing"
)

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func TestCostBillsCachedTokensAtCachedRate(t *testing.T) {
	p := Price{InputPerM: 0.30, OutputPerM: 2.50, CachedInputPerM: 0.03}
	if got := p.Cost(1_000_000, 1_000_000, 500_000); !approx(got, 0.5*0.30+0.5*0.03+2.50) {
		t.Fatalf("cost = %v", got)
	}
}

func TestReadsPythonLedgerFormat(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	// Lines exactly as maxionbench/harness/budget.py writes them.
	python := `{"at": "2026-10-05T00:00:00+00:00", "event": "reserve", "reservation_id": "a", "label": "x", "estimate_usd": 6.0}
{"at": "2026-10-05T00:00:01+00:00", "event": "commit", "reservation_id": "a", "label": "x", "actual_usd": 2.0, "usage": {"input_tokens": 100}}
{"at": "2026-10-05T00:00:02+00:00", "event": "reserve", "reservation_id": "b", "label": "y", "estimate_usd": 1.5}
`
	if err := os.WriteFile(path, []byte(python), 0o600); err != nil {
		t.Fatal(err)
	}
	l, err := Open(path, 10)
	if err != nil {
		t.Fatal(err)
	}
	rem, err := l.Remaining()
	if err != nil || !approx(rem, 10-2.0-1.5) {
		t.Fatalf("remaining = %v, %v", rem, err)
	}
}

func TestCapHoldsUnderConcurrency(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ledger.jsonl")
	l, _ := Open(path, 10)
	var wg sync.WaitGroup
	var mu sync.Mutex
	granted := 0
	for i := 0; i < 40; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			// Separate Ledger values exercise the flock path, not just the in-process mutex.
			other, _ := Open(path, 10)
			if _, err := other.Reserve(1.0, "c"); err == nil {
				mu.Lock()
				granted++
				mu.Unlock()
			} else if !errors.Is(err, ErrBudgetExceeded) {
				t.Errorf("unexpected error: %v", err)
			}
		}()
	}
	wg.Wait()
	if granted != 10 {
		t.Fatalf("granted %d reservations of $1 under a $10 cap", granted)
	}
	if rem, _ := l.Remaining(); !approx(rem, 0) {
		t.Fatalf("remaining = %v", rem)
	}
}

func TestCommitAndReleaseCloseReservations(t *testing.T) {
	l, _ := Open(filepath.Join(t.TempDir(), "ledger.jsonl"), 1)
	a, err := l.Reserve(0.6, "a")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := l.Reserve(0.5, "b"); !errors.Is(err, ErrBudgetExceeded) {
		t.Fatalf("expected ErrBudgetExceeded, got %v", err)
	}
	if err := l.Commit(a, 0.1, map[string]any{"input_tokens": 10}); err != nil {
		t.Fatal(err)
	}
	b, err := l.Reserve(0.9, "b")
	if err != nil {
		t.Fatalf("after commit 0.1, 0.9 should fit: %v", err)
	}
	if err := l.Release(b); err != nil {
		t.Fatal(err)
	}
	if rem, _ := l.Remaining(); !approx(rem, 0.9) {
		t.Fatalf("remaining = %v", rem)
	}
}
