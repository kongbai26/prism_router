# Security Policy

[English](SECURITY.md) | [简体中文](SECURITY_zh.md)

## 🛡️ Supported Versions

Security fixes and maintenance are provided for the following releases of Prism Router:

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

---

## 🚨 Reporting a Vulnerability

Security is critical for LLM proxy and gateway services. If you discover a security vulnerability in Prism Router (e.g., API key leakage risks, authentication bypass, SSRF, unauthorized endpoint access):

1. **Do NOT open a public issue.**
2. Please report the issue privately through [GitHub Private Vulnerability Reporting](https://github.com/kongbai26/prism_router/security/advisories/new).
3. In your report, please provide as much detail as possible:
   - Vulnerability type and affected component (e.g., `/v1/responses`, auth middleware, fallback chain, etc.)
   - Step-by-step reproduction instructions, a minimal Proof of Concept (PoC), or attack scenario description
   - Assessment of potential impact
   - Suggested remediation or patch, if available

### Response Timeline

- The maintainers will acknowledge and assess your report within 48 hours.
- Please collaborate with us under Responsible Disclosure principles until a patched release is published.
