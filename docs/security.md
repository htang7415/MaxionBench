# Security Policy

## Reporting

Report vulnerabilities privately through GitHub Security Advisories. Do not open a public issue for an undisclosed vulnerability.

Include the affected version, reproduction steps, impact, and any known mitigation. Do not include real credentials, private datasets, or personal data.

## Secrets and paid APIs

- API keys are supplied at runtime only: `GEMINI_API_KEY`, or a local key file (default
  `docs/gemini_api.txt`, override with `MAXIONBENCH_GEMINI_KEY_FILE`). Key files are excluded by
  explicit `.gitignore` and `.dockerignore` rules (`**/gemini_api.txt`, `*.key`, `.env*`).
- Keys are wrapped so `repr`, `str`, JSON, and pickle never reveal them. Error text bound for logs or
  result bundles is redacted, and provenance records only `gemini_key_present`.
- The Go gateway is the only process that holds the key when it fronts Gemini. It redacts the key
  from forwarded provider errors, never exports it in metrics, and its traces record routing
  attributes but no headers or bodies.
- Paid calls must reserve their estimated cost against a persistent ledger with a hard cap (default
  $10, `configs/pricing/gemini.yaml`) before they are sent; the cap holds across threads, processes,
  and the Python and Go components. Committed cost counts thinking tokens, which Gemini's
  OpenAI-compatible usage reports only in `total_tokens`.
- CI never calls paid APIs and has no secrets.

## Network exposure

Local services (gateway, engines, llm-d, observability stack) bind to 127.0.0.1. The llama.cpp and
vLLM servers have no API key; do not expose them beyond the host.

## Supported version

Security fixes target the latest release on `main`. Older benchmark artifacts remain immutable; fixes are published in a new release.
