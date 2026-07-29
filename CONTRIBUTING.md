# Contributing

1. Create a focused branch and keep behavior changes covered by tests.
2. Run `pytest -q` and `python -m compileall -q .`.
3. Keep Tianyan execution fail-fast: do not silently reorder, repair, or substitute cloud results.
4. Do not hard-code backend names, physical-qubit mappings, credentials, or user paths.
5. Update `CHANGELOG.md` for user-visible changes.
6. Preserve backward compatibility within the 1.x series unless a deprecation path is documented.
