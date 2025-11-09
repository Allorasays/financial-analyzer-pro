#!/bin/bash
# MONETA Backend Production Start Script for Render.com

echo "🚀 Starting MONETA Backend API..."
echo "Environment: $ENV"
echo "Port: ${PORT:-8000}"

# Start uvicorn with production settings
exec uvicorn proxy:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --log-level info \
    --timeout-keep-alive 30 \
    --no-access-log



