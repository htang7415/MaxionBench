// Package config loads the gateway configuration. Prices come from the same pricing file the Python
// harness uses (configs/pricing/gemini.yaml) so there is one source of truth for spend.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/yaml.v3"

	"github.com/htang7415/MaxionBench/gateway/internal/budget"
	"github.com/htang7415/MaxionBench/gateway/internal/ctxmgr"
)

// Policy values.
const (
	LocalOnly     = "local_only"
	LocalFirst    = "local_first"
	LocalFirstSLO = "local_first_slo" // overflow when the predicted local wait exceeds the TTFT SLO
	RemoteOnly    = "remote_only"
)

type Local struct {
	Upstreams   []string `yaml:"upstreams"`
	Model       string   `yaml:"model"`        // optional: rewrite the request's model field
	MaxInflight int      `yaml:"max_inflight"` // local_first overflows beyond this many in-flight requests
	TimeoutS    float64  `yaml:"timeout_s"`
	// local_first_slo: overflow when in-flight x recent per-request service time > SLOTTFTS.
	SLOTTFTS   float64 `yaml:"slo_ttft_s"`
	WindowS    float64 `yaml:"window_s"`    // service-time window
	MinSamples int     `yaml:"min_samples"` // completions needed before predicting; until then only MaxInflight applies
}

type Remote struct {
	Enabled         bool           `yaml:"enabled"`
	BaseURL         string         `yaml:"base_url"`
	ChatPath        string         `yaml:"chat_path"`
	Model           string         `yaml:"model"`
	KeyFile         string         `yaml:"key_file"` // used only if GEMINI_API_KEY is unset
	ReasoningEffort string         `yaml:"reasoning_effort"`
	PricingFile     string         `yaml:"pricing_file"`
	StripFields     []string       `yaml:"strip_fields"` // engine-specific fields the provider rejects
	TimeoutS        float64        `yaml:"timeout_s"`
	Price           budget.Price   `yaml:"-"`
	CapUSD          float64        `yaml:"-"`
	Extra           map[string]any `yaml:"extra_body"`
}

type Budget struct {
	LedgerPath string `yaml:"ledger_path"`
}

type Config struct {
	Listen   string `yaml:"listen"`
	Policy   string `yaml:"policy"`
	Failover bool   `yaml:"failover"` // retry on remote if local fails before any byte is sent
	Local    Local  `yaml:"local"`
	Remote   Remote `yaml:"remote"`
	Budget   Budget `yaml:"budget"`
	// Context is cache-aware context management for agent sessions (off by default).
	Context ctxmgr.Config `yaml:"context"`
}

type pricingFile struct {
	BudgetCapUSD float64                 `yaml:"budget_cap_usd"`
	Models       map[string]budget.Price `yaml:"models"`
}

// Load parses and validates the config, resolving prices and defaults.
func Load(path string) (*Config, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	cfg := &Config{
		Listen: "127.0.0.1:8090",
		Policy: LocalFirst,
		Local:  Local{MaxInflight: 8, TimeoutS: 300, WindowS: 10, MinSamples: 8},
		Remote: Remote{ChatPath: "/chat/completions", TimeoutS: 120,
			StripFields: []string{"chat_template_kwargs", "ignore_eos", "cache_prompt"}},
		Budget: Budget{LedgerPath: "~/.maxionbench/budget/gemini_ledger.jsonl"},
	}
	dec := yaml.NewDecoder(strings.NewReader(string(raw)))
	dec.KnownFields(true)
	if err := dec.Decode(cfg); err != nil {
		return nil, fmt.Errorf("%s: %w", path, err)
	}
	switch cfg.Policy {
	case LocalOnly, LocalFirst, LocalFirstSLO, RemoteOnly:
	default:
		return nil, fmt.Errorf("policy must be %s, %s, %s or %s", LocalOnly, LocalFirst, LocalFirstSLO, RemoteOnly)
	}
	if err := cfg.Context.Validate(); err != nil {
		return nil, err
	}
	if cfg.Policy == LocalFirstSLO && (cfg.Local.SLOTTFTS <= 0 || cfg.Local.WindowS <= 0 || cfg.Local.MinSamples < 1) {
		return nil, fmt.Errorf("policy %s needs local.slo_ttft_s > 0, window_s > 0 and min_samples >= 1", LocalFirstSLO)
	}
	if cfg.Policy != RemoteOnly && len(cfg.Local.Upstreams) == 0 {
		return nil, fmt.Errorf("local.upstreams is required for policy %s", cfg.Policy)
	}
	if cfg.Policy == RemoteOnly && !cfg.Remote.Enabled {
		return nil, fmt.Errorf("policy remote_only requires remote.enabled")
	}
	if cfg.Remote.Enabled {
		if cfg.Remote.BaseURL == "" || cfg.Remote.Model == "" || cfg.Remote.PricingFile == "" {
			return nil, fmt.Errorf("remote requires base_url, model and pricing_file")
		}
		pb, err := os.ReadFile(expand(cfg.Remote.PricingFile))
		if err != nil {
			return nil, err
		}
		var pf pricingFile
		if err := yaml.Unmarshal(pb, &pf); err != nil {
			return nil, err
		}
		price, ok := pf.Models[cfg.Remote.Model]
		if !ok {
			return nil, fmt.Errorf("no price for %q in %s; refusing to send paid requests", cfg.Remote.Model, cfg.Remote.PricingFile)
		}
		cfg.Remote.Price, cfg.Remote.CapUSD = price, pf.BudgetCapUSD
	}
	cfg.Budget.LedgerPath = expand(cfg.Budget.LedgerPath)
	return cfg, nil
}

// LoadKey returns the remote API key from GEMINI_API_KEY or the key file. Never log the result.
func LoadKey(r Remote) (string, error) {
	if k := strings.TrimSpace(os.Getenv("GEMINI_API_KEY")); k != "" {
		return k, nil
	}
	if r.KeyFile == "" {
		return "", fmt.Errorf("set GEMINI_API_KEY or remote.key_file")
	}
	b, err := os.ReadFile(expand(r.KeyFile))
	if err != nil {
		return "", fmt.Errorf("reading key file: %w", err)
	}
	k := strings.TrimSpace(string(b))
	if k == "" {
		return "", fmt.Errorf("key file is empty")
	}
	return k, nil
}

func expand(p string) string {
	if strings.HasPrefix(p, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			return filepath.Join(home, p[2:])
		}
	}
	return p
}
