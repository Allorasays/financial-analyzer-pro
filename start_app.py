#!/usr/bin/env python3
"""
Startup script for Financial Analyzer Pro on Render
This script helps with debugging and ensures proper startup
"""

import os
import sys
import subprocess

def main():
    print("🚀 Starting Financial Analyzer Pro...")
    
    # Get port from environment
    port = os.getenv("PORT", "8501")
    print(f"📡 Using port: {port}")
    
    # Get host from environment
    host = os.getenv("HOST", "0.0.0.0")
    print(f"🌐 Using host: {host}")
    
    # Check if app file exists
    app_file = "app_render_fixed.py"
    if not os.path.exists(app_file):
        print(f"❌ App file {app_file} not found!")
        sys.exit(1)
    
    print(f"✅ Found app file: {app_file}")
    
    # Start Streamlit
    cmd = [
        "streamlit", "run", app_file,
        "--server.port", port,
        "--server.address", host,
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"🔧 Running command: {' '.join(cmd)}")
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting app: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("🛑 App stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()