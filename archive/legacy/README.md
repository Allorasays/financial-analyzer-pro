# Legacy modules (archived)

These files are **not used** by the active deployment (`proxy.py` + `render_final.yaml`).
They contained bundled default API keys and duplicate aggregator logic.

Active replacements:
- `comprehensive_financial_aggregator.py` — financial metrics
- `config.py` — env-only API configuration

Do not import these archived modules in new code.
