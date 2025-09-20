#!/usr/bin/env python3
"""
Quick deployment script for Render - Simplified version
"""

import os
import shutil

def prepare_simple_deployment():
    """Prepare files for simple Render deployment"""
    print("🚀 Preparing simplified deployment for Render...")
    
    # Copy simplified files to main names
    files_to_copy = [
        ('app_render_simple.py', 'app.py'),
        ('requirements_render_simple.txt', 'requirements.txt'),
        ('render_simple.yaml', 'render.yaml'),
        ('Procfile_simple', 'Procfile')
    ]
    
    for source, target in files_to_copy:
        if os.path.exists(source):
            shutil.copy2(source, target)
            print(f"✅ Copied {source} → {target}")
        else:
            print(f"❌ Source file {source} not found")
    
    # Create runtime.txt
    with open('runtime.txt', 'w') as f:
        f.write('python-3.11.0\n')
    print("✅ Created runtime.txt")
    
    print("\n🎉 Simplified deployment files ready!")
    print("\n📋 Next steps:")
    print("1. Commit these files to GitHub:")
    print("   - app.py")
    print("   - requirements.txt") 
    print("   - render.yaml")
    print("   - Procfile")
    print("   - runtime.txt")
    print("\n2. Deploy on Render.com:")
    print("   - Create new Web Service")
    print("   - Connect GitHub repository")
    print("   - Use render.yaml configuration")
    print("   - Deploy!")

def test_simple_app():
    """Test the simplified app"""
    print("\n🧪 Testing simplified app...")
    
    try:
        # Test imports
        import streamlit as st
        import pandas as pd
        import plotly.graph_objects as go
        import yfinance as yf
        import numpy as np
        from sklearn.linear_model import LinearRegression
        print("✅ All imports successful")
        
        # Test app file
        with open('app_render_simple.py', 'r') as f:
            content = f.read()
            if 'def main():' in content and 'streamlit' in content:
                print("✅ App file structure looks good")
            else:
                print("❌ App file structure issues")
                return False
        
        print("✅ Simplified app test passed")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False

def main():
    """Main deployment preparation"""
    print("🔧 Render Deployment - Simplified Version")
    print("=" * 50)
    
    # Test the app first
    if not test_simple_app():
        print("❌ App test failed - fix issues first")
        return
    
    # Prepare deployment files
    prepare_simple_deployment()
    
    print("\n✅ Ready for Render deployment!")
    print("🚀 Your simplified Financial Analyzer Pro is ready to deploy!")

if __name__ == "__main__":
    main()



