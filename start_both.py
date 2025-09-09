#!/usr/bin/env python3
"""
Startup script that runs both FastAPI backend and Streamlit frontend
"""
import subprocess
import sys
import os
import time
import threading
import signal
import atexit

def run_fastapi():
    """Run FastAPI backend"""
    print("🚀 Starting FastAPI backend on port 8000...")
    try:
        subprocess.run([
            sys.executable, "proxy.py"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ FastAPI failed to start: {e}")
        sys.exit(1)

def run_streamlit():
    """Run Streamlit frontend"""
    print("🚀 Starting Streamlit frontend on port 8501...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--server.headless", "true",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Streamlit failed to start: {e}")
        sys.exit(1)

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down services...")
    sys.exit(0)

def main():
    """Main function to start both services"""
    print("🎯 Financial Analyzer Pro - Starting Services")
    print("=" * 50)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Register cleanup function
    atexit.register(lambda: print("🛑 Services stopped"))
    
    # Check if we're in production (Render)
    if os.getenv('RENDER'):
        print("🌐 Production mode detected - starting Streamlit only")
        run_streamlit()
    else:
        print("💻 Development mode - starting both services")
        
        # Start FastAPI in a separate thread
        fastapi_thread = threading.Thread(target=run_fastapi, daemon=True)
        fastapi_thread.start()
        
        # Wait a moment for FastAPI to start
        time.sleep(2)
        
        # Start Streamlit (main thread)
        run_streamlit()

if __name__ == "__main__":
    main()
