# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in Origin, **please do not open a public issue.**

Instead, email **h164654156465@gmail.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact

### Scope

We are especially interested in:

- Local privilege escalation via the HTTP API (`127.0.0.1:7878`)
- PII leaks through the memory pipeline or API responses
- Unauthorized access to stored memories or knowledge graph data
- Injection attacks (SQL, command, or prompt injection)

### Response

- We will acknowledge your report within **48 hours**
- We will provide a fix timeline within **7 days**
- We will credit you in the fix release (unless you prefer to remain anonymous)

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
