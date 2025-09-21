#!/usr/bin/env python3
"""
GameMatch Complete System Launcher
Starts both API server and User Dashboard with proper monitoring

Usage:
    python3 start_services.py          # Start both services
    python3 start_services.py --api     # Start only API
    python3 start_services.py --dash    # Start only Dashboard
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def start_api():
    """Start the API server"""
    print("🚀 Starting GameMatch API...")
    
    try:
        # Start API server
        api_process = subprocess.Popen([
            sys.executable, 'run_production_api.py'
        ], cwd=Path(__file__).parent)
        
        print(f"✅ API server started (PID: {api_process.pid})")
        return api_process
    except Exception as e:
        print(f"❌ Failed to start API: {e}")
        return None

def start_dashboard():
    """Start the dashboard"""
    print("🎨 Starting GameMatch Dashboard...")
    
    try:
        # Start dashboard
        dashboard_process = subprocess.Popen([
            sys.executable, 'src/web/dashboard.py'
        ], cwd=Path(__file__).parent)
        
        print(f"✅ Dashboard started (PID: {dashboard_process.pid})")
        return dashboard_process
    except Exception as e:
        print(f"❌ Failed to start dashboard: {e}")
        return None

def main():
    import sys
    
    print("🎮 GameMatch Complete System Launcher")
    print("=" * 50)
    
    # Parse command line arguments
    api_only = '--api' in sys.argv
    dash_only = '--dash' in sys.argv
    
    processes = []
    
    # Start services based on arguments
    if not dash_only:
        api_process = start_api()
        if api_process:
            processes.append(('API', api_process))
    
    # Wait a bit for API to start
    if not dash_only:
        time.sleep(3)
    
    if not api_only:
        dashboard_process = start_dashboard()
        if dashboard_process:
            processes.append(('Dashboard', dashboard_process))
    
    if not processes:
        print("❌ No services could be started")
        return
    
    print("\n" + "=" * 60)
    print("🎯 GameMatch System Status")
    print("=" * 60)
    
    if not dash_only:
        print("🔌 API Server: http://localhost:8000")
        print("📖 API Documentation: http://localhost:8000/docs")
        print("🔍 Health Check: http://localhost:8000/health")
    
    if not api_only:
        print("📊 User Dashboard: http://localhost:5000")
        print("🎮 Game Recommendations & Analytics")
    
    print(f"\n⚙️  Running {len(processes)} service(s)")
    print("💡 Press Ctrl+C to stop all services")
    print("=" * 60)
    
    # Wait for services to start
    time.sleep(5)
    
    # Check if services are still running
    for name, process in processes:
        if process.poll() is not None:
            print(f"⚠️  {name} service has stopped (return code: {process.returncode})")
    
    try:
        # Wait for interrupt
        while True:
            time.sleep(1)
            # Check if any process died
            for name, process in processes[:]:  # Copy list to avoid modification during iteration
                if process.poll() is not None:
                    print(f"⚠️  {name} service died with return code: {process.returncode}")
                    processes.remove((name, process))
            
            if not processes:
                print("❌ All processes have stopped")
                break
                
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        for name, process in processes:
            try:
                print(f"Stopping {name}...")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print(f"Force killing {name}...")
                process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main()