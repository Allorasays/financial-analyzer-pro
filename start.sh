#!/bin/bash
# Streamlit startup script for deployment
export STREAMLIT_SERVER_HEADLESS=true
export STREAMLIT_SERVER_PORT=${PORT:-8501}
export STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Wait for API to be ready (optional)
echo "Starting Streamlit application..."
echo "API Base URL: ${API_BASE_URL:-http://localhost:8000}"
echo "Server Port: ${STREAMLIT_SERVER_PORT}"
echo "Server Address: ${STREAMLIT_SERVER_ADDRESS}"

# Start Streamlit
streamlit run app.py \
  --server.port $STREAMLIT_SERVER_PORT \
  --server.address $STREAMLIT_SERVER_ADDRESS \
  --server.headless true \
  --server.enableCORS false \
  --server.enableXsrfProtection false

