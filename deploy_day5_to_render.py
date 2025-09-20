#!/usr/bin/env python3
"""
Deploy Day 5 Watchlist System to Render
This script helps prepare and deploy the Day 5 application to Render
"""

import os
import subprocess
import sys

def check_files():
    """Check if all required files exist"""
    required_files = [
        'app_day5_render.py',
        'requirements_render_day5.txt',
        'render_day5.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files found")
    return True

def check_dependencies():
    """Check if required packages are installed locally"""
    try:
        import streamlit
        import pandas
        import plotly
        import yfinance
        import numpy
        print("✅ All dependencies available locally")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        return False

def create_git_commit():
    """Create a git commit for deployment"""
    try:
        # Add files
        subprocess.run(['git', 'add', 'app_day5_render.py'], check=True)
        subprocess.run(['git', 'add', 'requirements_render_day5.txt'], check=True)
        subprocess.run(['git', 'add', 'render_day5.yaml'], check=True)
        subprocess.run(['git', 'add', 'RENDER_DAY5_DEPLOYMENT_GUIDE.md'], check=True)
        
        # Commit
        subprocess.run(['git', 'commit', '-m', 'Day 5: Watchlist System Enhanced for Render'], check=True)
        
        print("✅ Git commit created successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git commit failed: {e}")
        return False

def push_to_github():
    """Push changes to GitHub"""
    try:
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        print("✅ Changes pushed to GitHub")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Git push failed: {e}")
        return False

def main():
    """Main deployment process"""
    print("🚀 Day 5 Watchlist System - Render Deployment")
    print("=" * 50)
    
    # Check files
    if not check_files():
        print("❌ Please ensure all required files exist")
        return
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Please install missing dependencies")
        return
    
    # Create git commit
    if not create_git_commit():
        print("❌ Git commit failed")
        return
    
    # Push to GitHub
    if not push_to_github():
        print("❌ Git push failed")
        return
    
    print("\n🎉 Deployment preparation complete!")
    print("\n📋 Next Steps:")
    print("1. Go to render.com")
    print("2. Create new Web Service")
    print("3. Connect your GitHub repository")
    print("4. Use these settings:")
    print("   - Build Command: pip install -r requirements_render_day5.txt")
    print("   - Start Command: streamlit run app_day5_render.py --server.port $PORT --server.address 0.0.0.0")
    print("5. Set environment variables:")
    print("   - PYTHON_VERSION=3.11.0")
    print("   - STREAMLIT_SERVER_HEADLESS=true")
    print("   - STREAMLIT_SERVER_ENABLE_CORS=false")
    print("   - STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false")
    print("\n📖 See RENDER_DAY5_DEPLOYMENT_GUIDE.md for detailed instructions")

if __name__ == "__main__":
    main()
