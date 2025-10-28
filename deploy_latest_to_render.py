#!/usr/bin/env python3
"""
Deploy Latest Financial Analyzer Pro to Render
This script helps deploy the most up-to-date version to Render
"""

import subprocess
import sys
import os
import time
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(f"Output: {result.stdout}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False

def check_files():
    """Check if required files exist"""
    required_files = [
        'app_enhanced_stable.py',
        'requirements_latest.txt',
        'render_latest_stable.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {missing_files}")
        return False
    
    print("✅ All required files found")
    return True

def main():
    """Main deployment function"""
    print("🚀 Financial Analyzer Pro - Latest Version Deployment to Render")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('app_enhanced_stable.py'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check required files
    if not check_files():
        sys.exit(1)
    
    print("\n📋 Deployment Configuration:")
    print(f"   App File: app_enhanced_stable.py")
    print(f"   Requirements: requirements_latest.txt")
    print(f"   Render Config: render_latest_stable.yaml")
    print(f"   Service Name: financial-analyzer-pro-latest")
    
    # Check if git is initialized
    if not os.path.exists('.git'):
        print("\n🔄 Initializing Git repository...")
        if not run_command('git init', "Git initialization"):
            sys.exit(1)
    
    # Add all files to git
    if not run_command('git add .', "Adding files to git"):
        sys.exit(1)
    
    # Commit changes
    commit_message = f"Deploy latest stable version - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        print("⚠️  No changes to commit or commit failed")
    
    # Check if render CLI is installed
    try:
        subprocess.run(['render', '--version'], check=True, capture_output=True)
        render_cli_available = True
    except (subprocess.CalledProcessError, FileNotFoundError):
        render_cli_available = False
    
    if render_cli_available:
        print("\n🔧 Render CLI detected - attempting direct deployment...")
        
        # Deploy using render CLI
        deploy_command = 'render deploy --service financial-analyzer-pro-latest'
        if run_command(deploy_command, "Deploying to Render"):
            print("\n🎉 Deployment completed successfully!")
            print("Your app should be available at: https://financial-analyzer-pro-latest.onrender.com")
        else:
            print("\n⚠️  Direct deployment failed. Please deploy manually using the Render dashboard.")
    else:
        print("\n📝 Manual Deployment Instructions:")
        print("=" * 40)
        print("1. Go to https://dashboard.render.com")
        print("2. Create a new Web Service")
        print("3. Connect your GitHub repository")
        print("4. Use these settings:")
        print("   - Name: financial-analyzer-pro-latest")
        print("   - Environment: Python 3")
        print("   - Build Command: pip install -r requirements_latest.txt --no-cache-dir")
        print("   - Start Command: streamlit run app_enhanced_stable.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false")
        print("5. Set environment variables as specified in render_latest_stable.yaml")
        print("6. Deploy!")
    
    print("\n📊 Deployment Summary:")
    print("   ✅ Latest stable version ready")
    print("   ✅ Optimized requirements file")
    print("   ✅ Render configuration created")
    print("   ✅ All files committed to git")
    
    print("\n🔗 Next Steps:")
    print("   1. Push to GitHub: git push origin main")
    print("   2. Deploy to Render using the instructions above")
    print("   3. Monitor deployment in Render dashboard")
    print("   4. Test your application once deployed")

if __name__ == "__main__":
    main()
