#!/usr/bin/env python3
"""
Financial Analyzer Pro - Startup Script
This script helps you start both the FastAPI backend and Streamlit frontend.
"""

import subprocess
import sys
import time
import webbrowser
import os
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = ['streamlit', 'fastapi', 'uvicorn', 'pandas', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    
    try:
        # Start the backend server
        backend_process = subprocess.Popen([
            sys.executable, "proxy.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the server to start
        time.sleep(3)
        
        # Check if the server is running
        import requests
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print("✅ Backend server is running at http://localhost:8000")
                return backend_process
            else:
                print("❌ Backend server failed to start properly")
                return None
        except requests.exceptions.RequestException:
            print("❌ Backend server failed to start")
            return None
            
    except Exception as e:
        print(f"❌ Error starting backend: {e}")
        return None

def start_frontend():
    """Start the Streamlit frontend"""
    print("🌐 Starting Streamlit frontend...")
    
    try:
        # Start the Streamlit app
        frontend_process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        # Wait a moment for the app to start
        time.sleep(5)
        
        print("✅ Frontend is starting at http://localhost:8501")
        return frontend_process
        
    except Exception as e:
        print(f"❌ Error starting frontend: {e}")
        return None

def main():
    """Main startup function"""
    print("=" * 50)
    print("📊 Financial Analyzer Pro - Startup Script")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start backend
    backend_process = start_backend()
    if not backend_process:
        print("❌ Failed to start backend. Exiting.")
        sys.exit(1)
    
    # Start frontend
    frontend_process = start_frontend()
    if not frontend_process:
        print("❌ Failed to start frontend. Exiting.")
        backend_process.terminate()
        sys.exit(1)
    
    print("\n🎉 Both services are starting up!")
    print("\n📱 Access your application:")
    print("   Frontend: http://localhost:8501")
    print("   Backend API: http://localhost:8000")
    print("   API Docs: http://localhost:8000/docs")
    
    # Try to open the frontend in the browser
    try:
        webbrowser.open("http://localhost:8501")
        print("\n🌐 Opening frontend in your default browser...")
    except:
        print("\n🌐 Please manually open http://localhost:8501 in your browser")
    
    print("\n⏹️  Press Ctrl+C to stop both services")
    
    try:
        # Keep the script running
        while True:
            time.sleep(1)
            
            # Check if processes are still running
            if backend_process.poll() is not None:
                print("❌ Backend process stopped unexpectedly")
                break
            if frontend_process.poll() is not None:
                print("❌ Frontend process stopped unexpectedly")
                break
                
    except KeyboardInterrupt:
        print("\n\n🛑 Shutting down services...")
        
        # Terminate processes
        if backend_process:
            backend_process.terminate()
            print("✅ Backend stopped")
        
        if frontend_process:
            frontend_process.terminate()
            print("✅ Frontend stopped")
        
        print("👋 Goodbye!")

if __name__ == "__main__":
    main()

