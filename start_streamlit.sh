#!/bin/bash
# MONETA Web Dashboard Start Script for Render.com

echo "🚀 Starting MONETA Web Dashboard..."
echo "Port: ${PORT:-8501}"

# Start Streamlit with production settings
exec streamlit run app.py \
    --server.port=${PORT:-8501} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=true \
    --server.enableXsrfProtection=true


