# MONETA Production Deployment Review

Date: 2025-10-28
Owner: Deployment/Release

## Targets
- Backend: FastAPI (proxy.py), Render.com
- Web: Streamlit (app.py), Render.com
- Android: FinancialAnalyzerApp/ (native Kotlin)
- RN (Expo): FinancialAnalyzerMobile/

## Environment Variables
- Backend
  - PORT (Render-provided)
  - HOST=0.0.0.0
  - TIINGO_API_KEY, ALPHAVANTAGE_API_KEY, NEWSAPI_KEY, FRED_API_KEY, YF_PROXY_URL (optional)
  - ENABLE_TIINGO=true/false, ENABLE_ALPHA_VANTAGE=true/false
- Streamlit
  - STREAMLIT_SERVER_HEADLESS=true
  - STREAMLIT_SERVER_ADDRESS=0.0.0.0

## Health & Monitoring
- API status: GET /api/system/status (proxy)
- Streamlit dashboard: monitoring_dashboard.py

## Render Configuration
- Python: PYTHON_VERSION=3.11.0
- Build: pip install -r requirements.txt
- Start (backend): uvicorn proxy:app --host 0.0.0.0 --port $PORT
- Start (web): streamlit run app.py --server.port $PORT --server.address 0.0.0.0
- Networking: allow outbound HTTPS

## Scaling
- Start on Free; upgrade if p95 latency > 2s or rate limits frequent
- Enable autoscaling on backend (CPU target ~60%)

## Security
- Secrets via env vars only
- CORS: allow required origins
- Logs: redact keys/PII

## Data Providers
- Fallback: Yahoo → Tiingo → Alpha Vantage (FMP disabled)
- FRED attribution present in docs/listing

## Release Checklist
- [ ] Env vars set in Render (backend + web)
- [ ] Start commands configured
- [ ] /health and /api/system/status OK in prod
- [ ] Monitoring dashboard points to prod URL
- [ ] Android signed build tested
- [ ] Expo build tested (assets present)
- [ ] Privacy/Terms hosted at public URLs and linked

## Notes
- Keep financial_analyzer.db out of images; use managed storage if needed
- Prefer render_* files that start uvicorn/streamlit directly









