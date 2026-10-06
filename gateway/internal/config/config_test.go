package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func write(t *testing.T, dir, name, body string) string {
	t.Helper()
	p := filepath.Join(dir, name)
	if err := os.WriteFile(p, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	return p
}

func TestLoadResolvesPriceFromSharedPricingFile(t *testing.T) {
	dir := t.TempDir()
	pricing := write(t, dir, "gemini.yaml", "budget_cap_usd: 10.0\nmodels:\n  m1: {input_per_m: 0.3, output_per_m: 2.5, cached_input_per_m: 0.03}\n")
	cfg := write(t, dir, "gw.yaml", "policy: local_first\nlocal: {upstreams: [\"http://127.0.0.1:8081\"]}\n"+
		"remote: {enabled: true, base_url: \"https://x\", model: m1, pricing_file: \""+pricing+"\"}\n")
	c, err := Load(cfg)
	if err != nil {
		t.Fatal(err)
	}
	if c.Remote.Price.OutputPerM != 2.5 || c.Remote.CapUSD != 10 || c.Local.MaxInflight != 8 {
		t.Fatalf("loaded %+v", c.Remote)
	}
}

func TestLoadRejectsUnpricedModelUnknownFieldsAndBadPolicy(t *testing.T) {
	dir := t.TempDir()
	pricing := write(t, dir, "gemini.yaml", "budget_cap_usd: 10.0\nmodels: {}\n")
	cases := map[string]string{
		"no price":      "local: {upstreams: [\"http://a\"]}\nremote: {enabled: true, base_url: \"https://x\", model: m9, pricing_file: \"" + pricing + "\"}\n",
		"unknown field": "local: {upstreams: [\"http://a\"]}\nbogus: 1\n",
		"bad policy":    "policy: cheapest\nlocal: {upstreams: [\"http://a\"]}\n",
		"slo no target": "policy: local_first_slo\nlocal: {upstreams: [\"http://a\"]}\n",
		"no upstreams":  "policy: local_only\n",
	}
	for name, body := range cases {
		if _, err := Load(write(t, dir, strings.ReplaceAll(name, " ", "_")+".yaml", body)); err == nil {
			t.Errorf("%s: expected error", name)
		}
	}
}

func TestLoadKeyPrefersEnvAndNeverEchoesValue(t *testing.T) {
	dir := t.TempDir()
	keyFile := write(t, dir, "k.txt", "  from-file-key\n")
	t.Setenv("GEMINI_API_KEY", "")
	if k, err := LoadKey(Remote{KeyFile: keyFile}); err != nil || k != "from-file-key" {
		t.Fatalf("file key: %q %v", k, err)
	}
	t.Setenv("GEMINI_API_KEY", "from-env")
	if k, _ := LoadKey(Remote{KeyFile: keyFile}); k != "from-env" {
		t.Fatalf("env key not preferred: %q", k)
	}
	t.Setenv("GEMINI_API_KEY", "")
	empty := write(t, dir, "empty.txt", "\n")
	if _, err := LoadKey(Remote{KeyFile: empty}); err == nil {
		t.Fatal("empty key file accepted")
	}
}
