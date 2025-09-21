#!/usr/bin/env python3
"""
GameMatch Environment Setup Script
Automatically configures environment variables and validates the setup
"""

import os
import secrets
import sys
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GameMatchSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.env_file = self.project_root / '.env'
        self.env_example_file = self.project_root / '.env.example'
        
    def generate_secure_key(self, length=32):
        """Generate a secure random key"""
        return secrets.token_urlsafe(length)
    
    def create_env_example(self):
        """Create .env.example file with all required variables"""
        env_example_content = """# GameMatch Environment Configuration
# Copy this file to .env and fill in your actual values

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
MSSQL_SERVER=localhost
MSSQL_PORT=1433
MSSQL_DATABASE=gamematch
MSSQL_USERNAME=sa
MSSQL_PASSWORD=your_secure_password_here

# ===========================================
# FLASK WEB APPLICATION
# ===========================================
FLASK_SECRET_KEY=your_flask_secret_key_here
FLASK_PORT=5000
FLASK_HOST=127.0.0.1
DEBUG_MODE=false

# ===========================================
# API CONFIGURATION
# ===========================================
API_RATE_LIMIT=100
VALID_API_KEYS=gamematch-api-key-prod,gamematch-api-key-dev
DEFAULT_API_KEY=your_default_api_key_here

# ===========================================
# SECURITY SETTINGS
# ===========================================
ENVIRONMENT=development
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000,http://localhost:8000
JWT_SECRET_KEY=your_jwt_secret_key_here

# ===========================================
# PERFORMANCE TUNING
# ===========================================
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
STARTUP_DATASET_LIMIT=5000

# ===========================================
# OPENAI INTEGRATION
# ===========================================
OPENAI_API_KEY=your_openai_api_key_here

# ===========================================
# LOGGING & MONITORING
# ===========================================
LOG_LEVEL=INFO
"""
        
        with open(self.env_example_file, 'w') as f:
            f.write(env_example_content)
        
        logger.info(f"✅ Created {self.env_example_file}")
    
    def create_env_file(self, interactive=True):
        """Create .env file with secure defaults"""
        if self.env_file.exists():
            if interactive:
                response = input(f"⚠️  .env file already exists. Overwrite? (y/N): ")
                if response.lower() != 'y':
                    logger.info("Skipping .env file creation")
                    return
        
        # Generate secure keys
        flask_secret = self.generate_secure_key(32)
        jwt_secret = self.generate_secure_key(32)
        api_key_prod = f"gm-prod-{self.generate_secure_key(16)}"
        api_key_dev = f"gm-dev-{self.generate_secure_key(16)}"
        default_api_key = api_key_dev
        
        if interactive:
            print("\n🔧 Setting up GameMatch Environment Configuration")
            print("=" * 50)
            
            # Database configuration
            print("\n📊 Database Configuration:")
            mssql_server = input("MSSQL Server (localhost): ").strip() or "localhost"
            mssql_port = input("MSSQL Port (1433): ").strip() or "1433"
            mssql_database = input("Database name (gamematch): ").strip() or "gamematch"
            mssql_username = input("Username (sa): ").strip() or "sa"
            mssql_password = input("Password (required): ").strip()
            
            if not mssql_password:
                logger.error("❌ Database password is required!")
                return False
            
            # OpenAI API Key
            print("\n🤖 OpenAI Configuration:")
            openai_key = input("OpenAI API Key (optional): ").strip()
            
            # Environment
            print("\n🌍 Environment Configuration:")
            environment = input("Environment (development/production) [development]: ").strip() or "development"
            debug_mode = "true" if environment == "development" else "false"
            
            # Flask configuration
            flask_port = input("Flask Port (5000): ").strip() or "5000"
            flask_host = "0.0.0.0" if environment == "development" else "127.0.0.1"
            
        else:
            # Non-interactive defaults
            mssql_server = "localhost"
            mssql_port = "1433"
            mssql_database = "gamematch"
            mssql_username = "sa"
            mssql_password = "GameMatch2024!"  # Default for development
            openai_key = ""
            environment = "development"
            debug_mode = "true"
            flask_port = "5000"
            flask_host = "0.0.0.0"
        
        # Create .env content
        env_content = f"""# GameMatch Environment Configuration
# Generated on {os.popen('date').read().strip()}

# ===========================================
# DATABASE CONFIGURATION
# ===========================================
MSSQL_SERVER={mssql_server}
MSSQL_PORT={mssql_port}
MSSQL_DATABASE={mssql_database}
MSSQL_USERNAME={mssql_username}
MSSQL_PASSWORD={mssql_password}

# ===========================================
# FLASK WEB APPLICATION
# ===========================================
FLASK_SECRET_KEY={flask_secret}
FLASK_PORT={flask_port}
FLASK_HOST={flask_host}
DEBUG_MODE={debug_mode}

# ===========================================
# API CONFIGURATION
# ===========================================
API_RATE_LIMIT=100
VALID_API_KEYS={api_key_prod},{api_key_dev}
DEFAULT_API_KEY={default_api_key}

# ===========================================
# SECURITY SETTINGS
# ===========================================
ENVIRONMENT={environment}
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5000,http://localhost:8000
JWT_SECRET_KEY={jwt_secret}

# ===========================================
# PERFORMANCE TUNING
# ===========================================
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
STARTUP_DATASET_LIMIT=5000

# ===========================================
# OPENAI INTEGRATION
# ===========================================
OPENAI_API_KEY={openai_key}

# ===========================================
# LOGGING & MONITORING
# ===========================================
LOG_LEVEL=INFO
"""
        
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        
        logger.info(f"✅ Created {self.env_file}")
        
        if interactive:
            print(f"\n🔑 Generated API Keys:")
            print(f"   Production: {api_key_prod}")
            print(f"   Development: {api_key_dev}")
            print(f"\n💡 Save these keys securely!")
        
        return True
    
    def validate_setup(self):
        """Validate the current setup"""
        logger.info("🔍 Validating GameMatch setup...")
        
        issues = []
        
        # Check if .env file exists
        if not self.env_file.exists():
            issues.append("❌ .env file not found")
            return issues
        
        # Load environment variables
        try:
            from dotenv import load_dotenv
            load_dotenv(self.env_file)
        except ImportError:
            issues.append("⚠️  python-dotenv not installed (pip install python-dotenv)")
        
        # Check required variables
        required_vars = [
            'MSSQL_SERVER', 'MSSQL_PORT', 'MSSQL_DATABASE', 
            'MSSQL_USERNAME', 'MSSQL_PASSWORD',
            'FLASK_SECRET_KEY', 'ENVIRONMENT'
        ]
        
        for var in required_vars:
            if not os.getenv(var):
                issues.append(f"❌ Missing required variable: {var}")
        
        # Check database connection
        try:
            import pyodbc
            server = os.getenv('MSSQL_SERVER')
            port = os.getenv('MSSQL_PORT')
            database = os.getenv('MSSQL_DATABASE')
            username = os.getenv('MSSQL_USERNAME')
            password = os.getenv('MSSQL_PASSWORD')
            
            conn_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={server},{port};DATABASE={database};UID={username};PWD={password};TrustServerCertificate=yes"
            
            try:
                conn = pyodbc.connect(conn_string, timeout=5)
                conn.close()
                logger.info("✅ Database connection successful")
            except Exception as e:
                issues.append(f"⚠️  Database connection failed: {e}")
        
        except ImportError:
            issues.append("⚠️  pyodbc not installed for database testing")
        
        # Check file permissions
        if not os.access(self.env_file, os.R_OK):
            issues.append("❌ Cannot read .env file (permission issue)")
        
        return issues
    
    def install_dependencies(self):
        """Install required Python packages"""
        logger.info("📦 Installing required dependencies...")
        
        try:
            import subprocess
            
            # Install core requirements
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'python-dotenv>=1.0.0',
                'pyodbc>=4.0.39',
                'fastapi>=0.104.0',
                'flask>=3.0.0',
                'pandas>=2.1.0',
                'pydantic>=2.5.0'
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("✅ Dependencies installed successfully")
                return True
            else:
                logger.error(f"❌ Failed to install dependencies: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error installing dependencies: {e}")
            return False
    
    def run_setup(self, interactive=True):
        """Run complete setup process"""
        print("🚀 GameMatch Environment Setup")
        print("=" * 40)
        
        # Create .env.example
        self.create_env_example()
        
        # Create .env file
        if not self.create_env_file(interactive):
            return False
        
        # Validate setup
        issues = self.validate_setup()
        
        if issues:
            print("\n⚠️  Setup Issues Found:")
            for issue in issues:
                print(f"   {issue}")
            print("\n💡 Please resolve these issues before running the application")
        else:
            print("\n✅ Setup completed successfully!")
            print("\n🎯 Next Steps:")
            print("   1. Review your .env file settings")
            print("   2. Start the API: python run_production_api.py")
            print("   3. Start the dashboard: python run_dashboard.py")
            print("   4. Access the dashboard at: http://localhost:5000")
        
        return len(issues) == 0

def main():
    setup = GameMatchSetup()
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "validate":
            issues = setup.validate_setup()
            if issues:
                print("Setup Issues:")
                for issue in issues:
                    print(f"  {issue}")
                sys.exit(1)
            else:
                print("✅ Setup is valid!")
                sys.exit(0)
        
        elif command == "non-interactive":
            setup.run_setup(interactive=False)
        
        elif command == "install-deps":
            setup.install_dependencies()
        
        else:
            print("Usage: python setup_environment.py [validate|non-interactive|install-deps]")
            sys.exit(1)
    else:
        # Interactive setup
        setup.run_setup(interactive=True)

if __name__ == "__main__":
    main()