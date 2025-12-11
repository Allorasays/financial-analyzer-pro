#!/usr/bin/env python3
"""
Quick deployment script for Financial Analyzer Pro to Render.com
Prepares all necessary files and provides deployment instructions
"""

import os
import shutil
import subprocess
import sys

def check_required_files():
    """Check if all required files exist"""
    required_files = [
        'app.py',
        'realtime_data_service.py', 
        'realtime_dashboard.py',
        'websocket_service.py',
        'requirements.txt',
        'render_full_program.yaml'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing required files: {', '.join(missing_files)}")
        return False
    
    print("✅ All required files present")
    return True

def create_procfile():
    """Create Procfile for deployment"""
    procfile_content = """web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false
"""
    
    with open('Procfile', 'w') as f:
        f.write(procfile_content)
    print("✅ Created Procfile")

def create_runtime_txt():
    """Create runtime.txt with Python version"""
    runtime_content = """python-3.11.0
"""
    
    with open('runtime.txt', 'w') as f:
        f.write(runtime_content)
    print("✅ Created runtime.txt")

def copy_deployment_config():
    """Copy render_full_program.yaml to render.yaml for deployment"""
    if os.path.exists('render_full_program.yaml'):
        shutil.copy2('render_full_program.yaml', 'render.yaml')
        print("✅ Copied render_full_program.yaml to render.yaml")
    else:
        print("❌ render_full_program.yaml not found")

def check_git_status():
    """Check if this is a git repository"""
    try:
        result = subprocess.run(['git', 'status'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Git repository detected")
            return True
        else:
            print("⚠️ Not a git repository - you'll need to create one for GitHub deployment")
            return False
    except FileNotFoundError:
        print("⚠️ Git not found - you'll need to create a repository manually")
        return False

def show_deployment_instructions():
    """Show step-by-step deployment instructions"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    
    print("\n📋 STEP 1: Prepare Repository")
    print("- Create a GitHub repository")
    print("- Upload all files to the repository")
    print("- Ensure these files are included:")
    print("  • app.py (main application)")
    print("  • realtime_data_service.py")
    print("  • realtime_dashboard.py") 
    print("  • websocket_service.py")
    print("  • requirements.txt")
    print("  • render.yaml (deployment config)")
    print("  • Procfile")
    print("  • runtime.txt")
    
    print("\n📋 STEP 2: Deploy to Render")
    print("1. Go to https://render.com")
    print("2. Sign up/Login and connect GitHub")
    print("3. Click 'New +' → 'Blueprint'")
    print("4. Select your repository")
    print("5. Render will auto-detect render.yaml")
    print("6. Click 'Apply' to deploy")
    
    print("\n📋 STEP 3: Wait for Deployment")
    print("- Build time: 5-10 minutes")
    print("- Monitor build logs")
    print("- App will be available at: https://your-app-name.onrender.com")
    
    print("\n📋 STEP 4: Test Features")
    print("✅ Dashboard - Market overview")
    print("✅ Stock Analysis - Enter AAPL")
    print("✅ Global Markets - 12+ markets")
    print("✅ Real-Time Data - Live updates")
    print("✅ Enhanced ML - AI predictions")
    print("✅ Portfolio Management - Track investments")
    
    print("\n🎉 SUCCESS!")
    print("Your complete Financial Analyzer Pro will be live with all features!")

def main():
    """Main deployment preparation"""
    print("🔧 Financial Analyzer Pro - Render Deployment Preparation")
    print("="*60)
    
    # Check required files
    if not check_required_files():
        print("\n❌ Cannot proceed - missing required files")
        return
    
    # Create deployment files
    create_procfile()
    create_runtime_txt()
    copy_deployment_config()
    
    # Check git status
    is_git_repo = check_git_status()
    
    # Show instructions
    show_deployment_instructions()
    
    print("\n" + "="*60)
    print("✅ DEPLOYMENT PREPARATION COMPLETE!")
    print("="*60)
    
    if not is_git_repo:
        print("\n⚠️ IMPORTANT: Create a GitHub repository and upload all files")
        print("   Then follow the deployment instructions above")
    else:
        print("\n🚀 Ready to deploy! Follow the instructions above")

if __name__ == "__main__":
    main()
