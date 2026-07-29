# Security policy

## Credentials

Do not report or commit live Tianyan API keys. Revoke a key immediately if it appears in chat, shell history, logs, screenshots, issue trackers, or source control.

The examples read credentials from `TIANYAN_API_KEY` or the cqlib-tianyan credential store. They never print the credential value.

## Reporting vulnerabilities

Report vulnerabilities privately to the repository maintainers before opening a public issue. Include the affected version, reproduction steps, and impact. Do not include active credentials or private cloud responses.
