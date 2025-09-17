#!/usr/bin/env python3
"""
Startup script for Financial Analyzer Pro on Render
This script ensures proper startup and handles common deployment issues
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if all required packages are installed"""
    required_packages = ['streamlit', 'pandas', 'plotly', 'yfinance', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is available")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} is missing")
    
    if missing_packages:
        print(f"Installing missing packages: {missing_packages}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    
    return True

def start_streamlit():
    """Start the Streamlit application"""
    port = os.environ.get('PORT', '8501')
    
    # Get the app file
    app_file = 'app_minimal_deploy.py'
    if not os.path.exists(app_file):
        app_file = 'app.py'
    
    print(f"Starting Streamlit app: {app_file}")
    print(f"Port: {port}")
    
    # Build the command
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_file,
        "--server.port", port,
        "--server.address", "0.0.0.0",
        "--server.headless", "true",
        "--server.enableCORS", "false",
        "--server.enableXsrfProtection", "false"
    ]
    
    print(f"Command: {' '.join(cmd)}")
    
    try:
        # Start the process
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Streamlit started successfully")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"❌ Streamlit failed to start")
            print(f"STDOUT: {stdout.decode()}")
            print(f"STDERR: {stderr.decode()}")
            return None
            
    except Exception as e:
        print(f"❌ Error starting Streamlit: {e}")
        return None

def main():
    print("🚀 Starting Financial Analyzer Pro...")
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        sys.exit(1)
    
    # Start Streamlit
    process = start_streamlit()
    if process is None:
        print("❌ Failed to start application")
        sys.exit(1)
    
    # Keep the process running
    try:
        process.wait()
    except KeyboardInterrupt:
        print("🛑 Shutting down...")
        process.terminate()
        process.wait()

if __name__ == "__main__":
    main()







