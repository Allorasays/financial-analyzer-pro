#!/usr/bin/env python3
"""
Deployment script for Financial Analyzer Pro
This script handles the installation and startup process
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        "streamlit",
        "pandas", 
        "plotly",
        "yfinance",
        "numpy",
        "requests"
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")
            return False
    
    return True

def start_app():
    """Start the Streamlit app"""
    port = os.environ.get('PORT', '8501')
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true"
    ]
    
    print(f"Starting app on port {port}...")
    subprocess.run(cmd)

if __name__ == "__main__":
    print("🚀 Starting Financial Analyzer Pro deployment...")
    
    if install_requirements():
        print("✅ All packages installed successfully")
        start_app()
    else:
        print("❌ Package installation failed")
        sys.exit(1)






