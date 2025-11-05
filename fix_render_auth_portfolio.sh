#!/bin/bash
# Script to verify and fix authentication/portfolio issues on Render

echo "=========================================="
echo "Render Auth & Portfolio Fix Script"
echo "=========================================="
echo ""

echo "[1] Checking if SECRET_KEY is set in Render..."
echo "    Go to Render Dashboard > moneta-backend-api > Environment"
echo "    Add: SECRET_KEY = <random-secret-key>"
echo "    Or use: openssl rand -hex 32"
echo ""

echo "[2] Checking database initialization..."
echo "    Database should auto-initialize on startup"
echo "    Check logs for: 'Database initialized successfully'"
echo ""

echo "[3] Testing endpoints..."
echo "    Test: curl -X POST https://moneta-backend-api.onrender.com/api/auth/register"
echo "          -H 'Content-Type: application/json'"
echo "          -d '{\"username\":\"test\",\"email\":\"test@test.com\",\"password\":\"test123\"}'"
echo ""

echo "[4] Known Issues:"
echo "    - Database is ephemeral on Render free tier"
echo "    - Users must re-register after service restart"
echo "    - Portfolio data is lost on restart"
echo ""

echo "[5] Solutions:"
echo "    - Upgrade to PostgreSQL (recommended)"
echo "    - Upgrade to paid Render plan (persistent disk)"
echo "    - Or accept ephemeral database (current)"
echo ""

echo "=========================================="

