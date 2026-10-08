// Package ctxmgr is cache-aware context management in the request path. Agent clients resend their whole
// history every call; the manager keeps, per session, the view it last sent upstream and only appends to it
// (so the upstream prefix cache keeps hitting) until the view passes a token budget, then trims it once with
// a base policy (mask old tool results, or keep a window of recent exchanges). It mirrors
// maxionbench/agents/context.py CacheAware(Mask|Window); testdata/parity.json is generated from Python.
//
// A pause trigger trims early: after a session has been idle longer than the provider's cache lifetime the
// prefix is probably evicted anyway, so rewriting it costs nothing and shrinks every later call.
package ctxmgr

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"sync"
	"time"
	"unicode/utf8"
)

// MaskText matches context.py MASK_TEXT.
const MaskText = "[tool result omitted to save context; call the tool again if you need it]"

// Actions reported per request.
const (
	Append    = "append"     // the previous view plus the new messages
	Edit      = "edit"       // the appended view passed the budget: re-rendered by the base policy
	PauseEdit = "pause_edit" // idle past PauseS: re-rendered early
	Start     = "start"      // new or diverged session: rendered by the base policy
)

type Message = map[string]any

// Config selects the base policy; Policy "" or "off" disables the manager.
type Config struct {
	Policy       string  `yaml:"policy"`        // off, mask+cache, window+cache
	BudgetTokens int     `yaml:"budget_tokens"` // re-render when the appended view passes this
	Keep         int     `yaml:"keep"`          // exchanges kept by mask/window
	PauseS       float64 `yaml:"pause_s"`       // 0 disables the pause trigger
	MaxSessions  int     `yaml:"max_sessions"`  // least recently used sessions are dropped beyond this
}

// Validate fills defaults and checks the policy name.
func (c *Config) Validate() error {
	if c.BudgetTokens == 0 {
		c.BudgetTokens = 64_000
	}
	if c.MaxSessions == 0 {
		c.MaxSessions = 10_000
	}
	switch c.Policy {
	case "", "off":
		return nil
	case "mask+cache":
		if c.Keep == 0 {
			c.Keep = 4
		}
	case "window+cache":
		if c.Keep == 0 {
			c.Keep = 8
		}
	default:
		return fmt.Errorf("context.policy must be off, mask+cache or window+cache, got %q", c.Policy)
	}
	if c.BudgetTokens < 0 || c.Keep < 0 || c.PauseS < 0 || c.MaxSessions < 0 {
		return fmt.Errorf("context: budget_tokens, keep, pause_s and max_sessions must be >= 0")
	}
	return nil
}

// Enabled reports whether the manager rewrites anything.
func (c Config) Enabled() bool { return c.Policy != "" && c.Policy != "off" }

type session struct {
	seen []string  // hashes of the client messages reflected in prev
	prev []Message // last view sent upstream
	last time.Time
}

// Manager holds per-session state. Safe for concurrent use.
type Manager struct {
	cfg      Config
	mu       sync.Mutex
	sessions map[string]*session
}

func New(cfg Config) *Manager { return &Manager{cfg: cfg, sessions: map[string]*session{}} }

// Result describes one rewrite.
type Result struct {
	View      []Message
	Action    string
	TokensIn  int // estimate of the client's history
	TokensOut int // estimate of the view sent upstream
}

// Apply returns the view to send upstream for the client's full history.
func (m *Manager) Apply(key string, history []Message, now time.Time) Result {
	hashes := make([]string, len(history))
	for i, msg := range history {
		hashes[i] = hashMessage(msg)
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	s := m.sessions[key]
	if s == nil {
		m.evictIfFull()
		s = &session{}
		m.sessions[key] = s
	}
	res := Result{TokensIn: EstimateTokens(history)}
	base := m.base(history)
	switch {
	case s.prev == nil || !extends(hashes, s.seen):
		res.View, res.Action = base, Start
	case m.cfg.PauseS > 0 && now.Sub(s.last).Seconds() > m.cfg.PauseS:
		res.View, res.Action = base, PauseEdit
	default:
		appended := append(append([]Message{}, s.prev...), tail(base, len(history)-len(s.seen))...)
		if EstimateTokens(appended) <= m.cfg.BudgetTokens {
			res.View, res.Action = appended, Append
		} else {
			res.View, res.Action = base, Edit
		}
	}
	s.prev, s.seen, s.last = res.View, hashes, now
	res.TokensOut = EstimateTokens(res.View)
	return res
}

// SessionKey identifies an agent session: the client's prompt_cache_key if set, else its system message
// and task (the first two messages), which stay fixed for the whole session.
func SessionKey(promptCacheKey string, history []Message) string {
	if promptCacheKey != "" {
		return "k:" + promptCacheKey
	}
	h := sha256.New()
	for _, msg := range history[:min(2, len(history))] {
		h.Write([]byte(hashMessage(msg)))
	}
	return "h:" + string(h.Sum(nil))
}

// Sessions returns the number of tracked sessions.
func (m *Manager) Sessions() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.sessions)
}

func (m *Manager) evictIfFull() {
	if len(m.sessions) < m.cfg.MaxSessions {
		return
	}
	var oldest string
	var t time.Time
	for k, s := range m.sessions {
		if oldest == "" || s.last.Before(t) {
			oldest, t = k, s.last
		}
	}
	delete(m.sessions, oldest)
}

func (m *Manager) base(history []Message) []Message {
	head, exchanges := split(history)
	switch m.cfg.Policy {
	case "window+cache":
		if len(exchanges) > m.cfg.Keep {
			exchanges = exchanges[len(exchanges)-m.cfg.Keep:]
		}
		return flat(head, exchanges)
	default: // mask+cache
		old := len(exchanges) - m.cfg.Keep
		out := make([][]Message, len(exchanges))
		for i, ex := range exchanges {
			out[i] = ex
			if i >= old {
				continue
			}
			out[i] = make([]Message, len(ex))
			for j, msg := range ex {
				if msg["role"] == "tool" {
					msg = replaceContent(msg, MaskText)
				}
				out[i][j] = msg
			}
		}
		return flat(head, out)
	}
}

// split mirrors context.py split: head is the system message (if any) and the task; each exchange starts
// with an assistant message and holds what follows it.
func split(history []Message) ([]Message, [][]Message) {
	headLen := 1
	if len(history) > 1 && history[0]["role"] == "system" {
		headLen = 2
	}
	headLen = min(headLen, len(history))
	var exchanges [][]Message
	for _, msg := range history[headLen:] {
		if msg["role"] == "assistant" || len(exchanges) == 0 {
			exchanges = append(exchanges, []Message{msg})
		} else {
			exchanges[len(exchanges)-1] = append(exchanges[len(exchanges)-1], msg)
		}
	}
	return history[:headLen], exchanges
}

func flat(head []Message, exchanges [][]Message) []Message {
	out := append([]Message{}, head...)
	for _, ex := range exchanges {
		out = append(out, ex...)
	}
	return out
}

func replaceContent(msg Message, text string) Message {
	out := make(Message, len(msg))
	for k, v := range msg {
		if k != "tokens" {
			out[k] = v
		}
	}
	out["content"] = text
	return out
}

// tail is the last n messages of view (Python view[-n:] with view[-0:] read as empty).
func tail(view []Message, n int) []Message {
	if n <= 0 {
		return nil
	}
	return view[max(0, len(view)-n):]
}

func extends(hashes, seen []string) bool {
	if len(hashes) < len(seen) {
		return false
	}
	for i, h := range seen {
		if hashes[i] != h {
			return false
		}
	}
	return true
}

func hashMessage(msg Message) string {
	b, _ := json.Marshal(msg) // map keys are sorted, so equal messages hash equally
	sum := sha256.Sum256(b)
	return string(sum[:])
}

// EstimateTokens mirrors context.py estimate_tokens: ~4 characters per token plus 4 per message and per
// tool call. Non-string content counts its JSON encoding.
func EstimateTokens(msgs []Message) int {
	total := 0
	for _, msg := range msgs {
		if t, ok := msg["tokens"].(float64); ok {
			total += int(t)
		} else {
			total += chars(msg["content"])/4 + 4
		}
		calls, _ := msg["tool_calls"].([]any)
		for _, c := range calls {
			call, _ := c.(map[string]any)
			fn, _ := call["function"].(map[string]any)
			total += chars(fn["arguments"])/4 + 4
		}
	}
	return total
}

func chars(v any) int {
	switch x := v.(type) {
	case nil:
		return 0
	case string:
		return utf8.RuneCountInString(x)
	default:
		b, _ := json.Marshal(x)
		return utf8.RuneCount(b)
	}
}
