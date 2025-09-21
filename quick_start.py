#!/usr/bin/env python3
"""
GameMatch Quick Start Script
One-command setup and launch for GameMatch system
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path
import logging
from threading import Thread
import webbrowser

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameMatchLauncher:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.processes = []
        self.running = True
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("🛑 Shutdown signal received...")
        self.running = False
        self.cleanup()
        sys.exit(0)
    
    def cleanup(self):
        """Clean up running processes"""
        logger.info("🧹 Cleaning up processes...")
        for process in self.processes:
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            except Exception as e:
                logger.warning(f"Error terminating process: {e}")
    
    def check_dependencies(self):
        """Check if required dependencies are installed"""
        logger.info("🔍 Checking dependencies...")
        
        required_packages = [
            'flask', 'fastapi', 'pandas', 'python-dotenv', 'pydantic'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package.replace('-', '_'))
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.warning(f"⚠️  Missing packages: {', '.join(missing)}")
            response = input("Install missing packages? (Y/n): ")
            if response.lower() != 'n':
                return self.install_dependencies(missing)
            return False
        
        logger.info("✅ All dependencies are installed")
        return True
    
    def install_dependencies(self, packages=None):
        """Install required packages"""
        if packages is None:
            packages = ['python-dotenv', 'flask', 'fastapi', 'pandas', 'pydantic', 'pyodbc']
        
        logger.info(f"📦 Installing packages: {', '.join(packages)}")
        
        try:
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install'
            ] + packages, check=True, capture_output=True, text=True)
            
            logger.info("✅ Dependencies installed successfully")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ Failed to install dependencies: {e.stderr}")
            return False
    
    def setup_environment(self):
        """Setup environment configuration"""
        env_file = self.project_root / '.env'
        
        if not env_file.exists():
            logger.info("🔧 Setting up environment configuration...")
            
            # Run setup script
            setup_script = self.project_root / 'setup_environment.py'
            if setup_script.exists():
                try:
                    result = subprocess.run([
                        sys.executable, str(setup_script), 'non-interactive'
                    ], check=True, capture_output=True, text=True)
                    logger.info("✅ Environment setup completed")
                    return True
                except subprocess.CalledProcessError as e:
                    logger.error(f"❌ Environment setup failed: {e}")
                    return False
            else:
                logger.error("❌ Setup script not found")
                return False
        else:
            logger.info("✅ Environment configuration already exists")
            return True
    
    def validate_setup(self):
        """Validate the current setup"""
        logger.info("🔍 Validating setup...")
        
        # Check .env file
        env_file = self.project_root / '.env'
        if not env_file.exists():
            logger.error("❌ .env file not found")
            return False
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(env_file)
        except ImportError:
            logger.error("❌ python-dotenv not available")
            return False
        
        # Check required variables
        required_vars = ['MSSQL_PASSWORD', 'FLASK_SECRET_KEY']
        for var in required_vars:
            if not os.getenv(var):
                logger.error(f"❌ Missing required variable: {var}")
                return False
        
        logger.info("✅ Setup validation passed")
        return True
    
    def start_api(self):
        """Start the FastAPI server"""
        logger.info("🚀 Starting GameMatch API...")
        
        api_script = self.project_root / 'run_production_api.py'
        if not api_script.exists():
            # Try alternative locations
            api_script = self.project_root / 'src' / 'api' / 'main.py'
        
        if api_script.exists():
            try:
                process = subprocess.Popen([
                    sys.executable, str(api_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                self.processes.append(process)
                logger.info("✅ API server started")
                return process
            except Exception as e:
                logger.error(f"❌ Failed to start API: {e}")
                return None
        else:
            logger.warning("⚠️  API script not found, skipping API server")
            return None
    
    def start_dashboard(self):
        """Start the Flask dashboard"""
        logger.info("🎨 Starting GameMatch Dashboard...")
        
        dashboard_script = self.project_root / 'run_dashboard.py'
        if not dashboard_script.exists():
            # Try alternative locations
            dashboard_script = self.project_root / 'src' / 'web' / 'dashboard.py'
        
        if dashboard_script.exists():
            try:
                process = subprocess.Popen([
                    sys.executable, str(dashboard_script)
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                self.processes.append(process)
                logger.info("✅ Dashboard started")
                return process
            except Exception as e:
                logger.error(f"❌ Failed to start dashboard: {e}")
                return None
        else:
            logger.warning("⚠️  Dashboard script not found, skipping dashboard")
            return None
    
    def wait_for_services(self):
        """Wait for services to be ready"""
        logger.info("⏳ Waiting for services to start...")
        
        # Wait a bit for services to initialize
        time.sleep(3)
        
        # Check if services are responding
        try:
            import requests
            
            # Check API
            try:
                response = requests.get('http://localhost:8000/health', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ API is ready")
                else:
                    logger.warning("⚠️  API health check failed")
            except:
                logger.warning("⚠️  API not responding")
            
            # Check Dashboard
            try:
                response = requests.get('http://localhost:5000', timeout=5)
                if response.status_code == 200:
                    logger.info("✅ Dashboard is ready")
                else:
                    logger.warning("⚠️  Dashboard health check failed")
            except:
                logger.warning("⚠️  Dashboard not responding")
                
        except ImportError:
            logger.info("📦 Install 'requests' package for health checks")
    
    def open_browser(self):
        """Open browser to dashboard"""
        try:
            dashboard_url = f"http://localhost:{os.getenv('FLASK_PORT', '5000')}"
            logger.info(f"🌐 Opening browser to {dashboard_url}")
            webbrowser.open(dashboard_url)
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")
    
    def monitor_processes(self):
        """Monitor running processes"""
        while self.running:
            for i, process in enumerate(self.processes):
                if process.poll() is not None:
                    logger.warning(f"⚠️  Process {i} has stopped")
                    # Optionally restart the process here
            
            time.sleep(5)
    
    def show_status(self):
        """Show current system status"""
        print("\n" + "="*50)
        print("🎮 GameMatch System Status")
        print("="*50)
        
        # Environment info
        print(f"📁 Project Root: {self.project_root}")
        print(f"🌍 Environment: {os.getenv('ENVIRONMENT', 'development')}")
        print(f"🐛 Debug Mode: {os.getenv('DEBUG_MODE', 'false')}")
        
        # Service URLs
        api_port = os.getenv('API_PORT', '8000')
        dashboard_port = os.getenv('FLASK_PORT', '5000')
        
        print(f"\n🔗 Service URLs:")
        print(f"   📊 Dashboard: http://localhost:{dashboard_port}")
        print(f"   🔌 API: http://localhost:{api_port}")
        print(f"   📖 API Docs: http://localhost:{api_port}/docs")
        
        # Process status
        print(f"\n⚙️  Running Processes: {len(self.processes)}")
        
        print("\n💡 Commands:")
        print("   Ctrl+C: Stop all services")
        print("   Check logs in terminal for detailed information")
        print("="*50)
    
    def run(self):
        """Run the complete GameMatch system"""
        print("🚀 GameMatch Quick Start")
        print("="*30)
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Dependency check failed")
            return False
        
        # Step 2: Setup environment
        if not self.setup_environment():
            logger.error("❌ Environment setup failed")
            return False
        
        # Step 3: Validate setup
        if not self.validate_setup():
            logger.error("❌ Setup validation failed")
            return False
        
        # Step 4: Start services
        api_process = self.start_api()
        dashboard_process = self.start_dashboard()
        
        if not api_process and not dashboard_process:
            logger.error("❌ No services could be started")
            return False
        
        # Step 5: Wait for services
        self.wait_for_services()
        
        # Step 6: Show status
        self.show_status()
        
        # Step 7: Open browser
        if dashboard_process:
            time.sleep(2)
            self.open_browser()
        
        # Step 8: Monitor processes
        try:
            logger.info("🎯 GameMatch is running! Press Ctrl+C to stop.")
            self.monitor_processes()
        except KeyboardInterrupt:
            logger.info("🛑 Shutdown requested by user")
        finally:
            self.cleanup()
        
        return True

def main():
    launcher = GameMatchLauncher()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "setup-only":
            launcher.check_dependencies()
            launcher.setup_environment()
            launcher.validate_setup()
            print("✅ Setup completed. Run without arguments to start services.")
        
        elif command == "validate":
            if launcher.validate_setup():
                print("✅ Setup is valid!")
                sys.exit(0)
            else:
                print("❌ Setup validation failed!")
                sys.exit(1)
        
        elif command == "install-deps":
            launcher.install_dependencies()
        
        else:
            print("Usage: python quick_start.py [setup-only|validate|install-deps]")
            sys.exit(1)
    else:
        # Full launch
        success = launcher.run()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()