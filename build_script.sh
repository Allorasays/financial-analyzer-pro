#!/bin/bash
echo "🚀 Starting Financial Analyzer Pro build..."

# Upgrade pip and install build tools
echo "📦 Upgrading pip and installing build tools..."
pip install --upgrade pip
pip install --upgrade setuptools wheel

# Install requirements
echo "📋 Installing requirements..."
pip install -r requirements_ultra_minimal.txt --no-cache-dir

# Verify installation
echo "✅ Verifying installation..."
python -c "import streamlit; import pandas; import yfinance; import numpy; print('All packages imported successfully!')"

echo "🎉 Build completed successfully!"









































