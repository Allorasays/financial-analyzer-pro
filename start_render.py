#!/usr/bin/env python3
"""
Startup script for Financial Analyzer Pro on Render
Ensures proper initialization and error handling
"""

import os
import sys
import subprocess

def main():
    # Set environment variables for Render
    os.environ['STREAMLIT_SERVER_HEADLESS'] = 'true'
    os.environ['STREAMLIT_SERVER_ADDRESS'] = '0.0.0.0'
    os.environ['STREAMLIT_SERVER_ENABLE_CORS'] = 'false'
    os.environ['STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION'] = 'false'
    os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'
    os.environ['STREAMLIT_SERVER_RUN_ON_SAVE'] = 'false'
    
    # Get port from environment (Render sets this)
    port = os.environ.get('PORT', '8501')
    
    # Start streamlit with proper configuration
    cmd = [
        sys.executable, '-m', 'streamlit', 'run', 'app.py',
        '--server.port', port,
        '--server.address', '0.0.0.0',
        '--server.headless', 'true',
        '--server.enableCORS', 'false',
        '--server.enableXsrfProtection', 'false',
        '--server.runOnSave', 'false',
        '--browser.gatherUsageStats', 'false'
    ]
    
    print(f"Starting Financial Analyzer Pro on port {port}")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the application
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error starting application: {e}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("Application stopped by user")
        sys.exit(0)

if __name__ == "__main__":
    main()