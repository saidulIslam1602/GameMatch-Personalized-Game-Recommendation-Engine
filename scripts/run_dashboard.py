#!/usr/bin/env python3
"""
Run GameMatch Visual Dashboard
Industry-standard web interface for game recommendations
"""

import os
import sys
import subprocess
import webbrowser
import time
import signal
import threading
from pathlib import Path

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    print("\n👋 Shutting down GameMatch Dashboard...")
    sys.exit(0)

def main():
    """Run the GameMatch dashboard with full industry standards"""
    print("🎮 GAMEMATCH VISUAL DASHBOARD")
    print("=" * 50)
    
    # Set up signal handling
    signal.signal(signal.SIGINT, signal_handler)
    
    # Get project root and set up paths
    project_root = Path(__file__).parent.parent
    src_path = project_root / "src"
    
    # Add src to Python path
    sys.path.insert(0, str(src_path))
    
    # Change to project root directory
    os.chdir(project_root)
    
    print(f"📁 Project root: {project_root}")
    print(f"🐍 Python path: {sys.path[:3]}")
    
    # Check dependencies
    required_packages = ['flask', 'flask-cors', 'pandas', 'numpy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✅ {package} is installed")
        except ImportError:
            missing_packages.append(package)
            print(f"❌ {package} not found")
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, "-m", "pip", "install"] + missing_packages, check=True)
        print("✅ All packages installed successfully")
    
    # Check if we have processed data
    data_file = project_root / "data" / "processed" / "steam_games_processed.parquet"
    if not data_file.exists():
        print("❌ Processed data not found. Running data preprocessing...")
        try:
            subprocess.run([sys.executable, "src/data/dataset_loader.py"], cwd=project_root, check=True)
            print("✅ Data preprocessing completed")
        except subprocess.CalledProcessError as e:
            print(f"❌ Data preprocessing failed: {e}")
            print("💡 Please run manually: python3 src/data/dataset_loader.py")
            return
    else:
        print(f"✅ Processed data found: {data_file}")
        print(f"📊 Data size: {data_file.stat().st_size / (1024*1024):.1f} MB")
    
    # Start the dashboard
    print("\n🚀 Starting GameMatch Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 Features:")
    print("   • Interactive game recommendations")
    print("   • Visual statistics and charts")
    print("   • Game details and images")
    print("   • Real-time search and filtering")
    print("   • Industry-standard API endpoints")
    print("   • Health monitoring and status checks")
    print("\n💡 Try searching for:")
    print("   • 'action games'")
    print("   • 'strategy'")
    print("   • 'indie'")
    print("   • 'multiplayer'")
    print("   • 'puzzle'")
    print("\n⏳ Opening dashboard in your browser...")
    
    # Open browser after a short delay
    def open_browser():
        time.sleep(3)
        try:
            webbrowser.open('http://localhost:5000')
            print("🌐 Dashboard opened in browser")
        except Exception as e:
            print(f"⚠️  Could not open browser: {e}")
            print("💡 Please manually open: http://localhost:5000")
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Run the dashboard with full error handling
    try:
        # Import the dashboard app
        from web.dashboard import app
        
        print("\n🚀 Starting GameMatch Dashboard Server...")
        print("🌐 Dashboard: http://localhost:5000")
        print("📊 Health Check: http://localhost:5000/health")
        print("🔧 API Status: http://localhost:5000/api/status")
        print("📋 API Docs: http://localhost:5000/api/recommendations?query=test")
        print("\n⏹️  Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Run with production-ready settings
        app.run(
            debug=False, 
            host='0.0.0.0', 
            port=5000, 
            threaded=True,
            use_reloader=False
        )
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure all dependencies are installed:")
        print("   pip install flask flask-cors pandas numpy")
        print("💡 Check that the src/web/dashboard.py file exists")
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")
        print("💡 Debug information:")
        print(f"   Project root: {project_root}")
        print(f"   Current dir: {os.getcwd()}")
        print(f"   Python path: {sys.path[:3]}")
        print("💡 Try running manually:")
        print("   cd /path/to/project")
        print("   python3 -c 'from src.web.dashboard import app; app.run()'")

if __name__ == "__main__":
    main()