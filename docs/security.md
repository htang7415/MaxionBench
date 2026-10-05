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
- Paid calls must reserve their estimated cost against a persistent ledger with a hard cap (default
  $10, `configs/pricing/gemini.yaml`); the cap holds across threads and processes.
- CI never calls paid APIs.

## Supported version

Security fixes target the latest release on `main`. Older benchmark artifacts remain immutable; fixes are published in a new release.
