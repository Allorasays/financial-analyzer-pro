#!/bin/bash
# Backend startup script for Render
set -e

echo "Starting MONETA Backend API..."

# Activate Python environment
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Start uvicorn using Python module
python -m uvicorn proxy:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --proxy-headers \
    --forwarded-allow-ips="*" \
    --log-level info

