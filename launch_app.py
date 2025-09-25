#!/usr/bin/env python3
"""
Launch the Financial Analyzer Pro with all Day 1-8 features
"""

import subprocess
import sys
import os

def launch_app():
    """Launch the Streamlit app"""
    print("🚀 Launching Financial Analyzer Pro...")
    print("=" * 50)
    print("📋 Available Features:")
    features = [
        "🏠 Dashboard",
        "📊 Stock Analysis", 
        "💼 Portfolio Management",
        "📈 Market Overview",
        "🔴 Real-Time Data",
        "🏭 Industry Analysis",
        "⚠️ Risk Assessment",
        "🤖 Enhanced ML",
        "📊 Technical Analysis",
        "📤 Export & Reports",
        "⚙️ Settings"
    ]
    
    for feature in features:
        print(f"   ✅ {feature}")
    
    print("\n🌐 Starting Streamlit server...")
    print("📱 The app will open in your browser at: http://localhost:8501")
    print("🔄 Use the sidebar to navigate between features")
    print("\n" + "=" * 50)
    
    try:
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "app.py",
            "--server.port", "8501",
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ])
    except KeyboardInterrupt:
        print("\n👋 App closed by user")
    except Exception as e:
        print(f"❌ Error launching app: {str(e)}")
        print("\n🔧 Alternative launch methods:")
        print("   1. Run: python -m streamlit run app.py")
        print("   2. Run: streamlit run app.py")
        print("   3. Check if Streamlit is properly installed")

if __name__ == "__main__":
    launch_app()

