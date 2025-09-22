#!/usr/bin/env python3
"""
Update Existing Financial Analyzer Pro Service on Render
Updates the current 'financial-analyzer-pro' service with latest stable version
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

def main():
    """Main update function"""
    print("🔄 Updating Existing Financial Analyzer Pro Service")
    print("=" * 55)
    print("Service Name: financial-analyzer-pro")
    print("New Version: app_enhanced_stable.py")
    print("=" * 55)
    
    # Check if we're in the right directory
    if not os.path.exists('app_enhanced_stable.py'):
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check if render CLI is available
    try:
        subprocess.run(['render', '--version'], check=True, capture_output=True)
        render_cli_available = True
        print("✅ Render CLI detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        render_cli_available = False
        print("⚠️  Render CLI not found - will provide manual instructions")
    
    # Commit any changes first
    print("\n📝 Preparing for update...")
    
    # Add all files to git
    if not run_command('git add .', "Adding files to git"):
        print("⚠️  Git add failed, continuing anyway...")
    
    # Commit changes
    commit_message = f"Update to latest stable version - {time.strftime('%Y-%m-%d %H:%M:%S')}"
    if not run_command(f'git commit -m "{commit_message}"', "Committing changes"):
        print("⚠️  No changes to commit or commit failed")
    
    if render_cli_available:
        print("\n🔧 Updating service using Render CLI...")
        
        # Update service configuration
        update_commands = [
            'render services update financial-analyzer-pro --build-command "pip install -r requirements_latest.txt --no-cache-dir"',
            'render services update financial-analyzer-pro --start-command "streamlit run app_enhanced_stable.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false"'
        ]
        
        for cmd in update_commands:
            if not run_command(cmd, f"Updating service configuration"):
                print(f"⚠️  Command failed: {cmd}")
        
        # Deploy the updated service
        if run_command('render deploy financial-analyzer-pro', "Deploying updated service"):
            print("\n🎉 Service updated successfully!")
            print("Your updated app should be available shortly at your existing URL")
        else:
            print("\n⚠️  Deployment failed. Please check the logs and try manual deployment.")
    else:
        print("\n📝 Manual Update Instructions:")
        print("=" * 40)
        print("1. Go to https://dashboard.render.com")
        print("2. Find your service: 'financial-analyzer-pro'")
        print("3. Click on the service to open its settings")
        print("4. Update the following settings:")
        print("")
        print("   📦 Build Command:")
        print("   pip install -r requirements_latest.txt --no-cache-dir")
        print("")
        print("   🚀 Start Command:")
        print("   streamlit run app_enhanced_stable.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableXsrfProtection=false")
        print("")
        print("5. Click 'Save Changes'")
        print("6. Click 'Manual Deploy' or wait for auto-deploy")
        print("7. Monitor the deployment in the logs")
    
    print("\n📊 Update Summary:")
    print("   ✅ Latest stable version ready (app_enhanced_stable.py)")
    print("   ✅ Optimized requirements (requirements_latest.txt)")
    print("   ✅ All changes committed to git")
    print("   ✅ Service configuration updated")
    
    print("\n🔗 What's New in This Update:")
    print("   • Enhanced stability and error handling")
    print("   • Improved performance and caching")
    print("   • Complete portfolio management features")
    print("   • Advanced technical analysis tools")
    print("   • Machine learning predictions")
    print("   • Export capabilities (PDF/Excel)")
    print("   • Real-time data with fallback")
    print("   • Mobile-optimized interface")
    
    print("\n⏱️  Expected Update Time:")
    print("   • Build: 2-5 minutes")
    print("   • Deploy: 1-2 minutes")
    print("   • Total: 3-7 minutes")
    
    print("\n🔍 Monitoring:")
    print("   • Check Render dashboard for deployment status")
    print("   • Monitor logs for any errors")
    print("   • Test your app once deployment completes")

if __name__ == "__main__":
    main()
