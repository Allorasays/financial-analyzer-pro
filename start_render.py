#!/usr/bin/env python3
"""
Render deployment startup script for Financial Analyzer Pro
This script ensures proper startup on Render's platform
"""

import os
import subprocess
import sys

def main():
    """Start the Streamlit app with proper configuration for Render"""
    
    # Get port from environment variable (Render sets this)
    port = os.environ.get('PORT', '8501')
    
    # Set Streamlit configuration
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    
    print(f"🚀 Starting Financial Analyzer Pro on port {port}")
    print("📊 Streamlit configuration:")
    print(f"   - Server Address: 0.0.0.0")
    print(f"   - Server Port: {port}")
    print(f"   - Headless: true")
    print(f"   - CORS: false")
    print(f"   - XSRF Protection: false")
    
    # Start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting Streamlit: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down...")
        sys.exit(0)

if __name__ == "__main__":
    main()



