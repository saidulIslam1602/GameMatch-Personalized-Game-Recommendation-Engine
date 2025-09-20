#!/usr/bin/env python3
"""
GameMatch Production API Runner
Launch the complete FastAPI microservice with all enhanced features
"""

import sys
import os
import subprocess
import time
import requests
from pathlib import Path
import json
import logging

# Add src to path
sys.path.append("src")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Check if all required components are available"""
    
    print("🔍 Checking system dependencies...")
    
    checks = {
        "Steam Dataset": Path("data/processed/steam_games_processed.parquet").exists(),
        "MSSQL Config": Path("config/mssql_config.json").exists(),
        "OpenAI Key": Path("config/openai_key.txt").exists(),
    }
    
    for check_name, status in checks.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {check_name}")
    
    if not all(checks.values()):
        print("\n⚠️ Some dependencies are missing. API will run with limited functionality.")
    else:
        print("\n🎉 All dependencies available!")
    
    return checks

def test_sql_connection():
    """Test SQL Server connection"""
    
    print("🔗 Testing SQL Server connection...")
    
    try:
        from models.mlops_monitoring import RecommendationTracker
        tracker = RecommendationTracker()
        
        # Try a simple operation
        test_id = tracker.log_recommendation(
            query="test connection",
            user_id="system_test",
            game_ids=[1, 2, 3],
            response_time_ms=100.0,
            confidence_score=0.5,
            model_version="GameMatch-System-Test"
        )
        
        print(f"   ✅ SQL Server connection successful (Test ID: {test_id})")
        return True
        
    except Exception as e:
        print(f"   ❌ SQL Server connection failed: {e}")
        print("   ⚠️ API will run without database logging")
        return False

def start_api_server(port=8000, reload=True):
    """Start the FastAPI server"""
    
    print(f"🚀 Starting GameMatch API server on port {port}...")
    
    try:
        # Import here to check for import errors
        import uvicorn
        from src.api.main import app
        
        print("   ✅ FastAPI application loaded successfully")
        
        # Start server
        uvicorn.run(
            "src.api.main:app",
            host="0.0.0.0",
            port=port,
            reload=reload,
            log_level="info"
        )
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("   💡 Try: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"   ❌ Server startup failed: {e}")
        return False

def test_api_endpoints(base_url="http://localhost:8000"):
    """Test API endpoints after server starts"""
    
    print(f"🧪 Testing API endpoints at {base_url}...")
    
    # Wait for server to start
    print("   ⏳ Waiting for server to start...")
    time.sleep(3)
    
    tests = [
        {
            "name": "Health Check",
            "method": "GET",
            "url": f"{base_url}/health",
            "expected_status": 200
        },
        {
            "name": "System Stats",
            "method": "GET", 
            "url": f"{base_url}/analytics/stats",
            "headers": {"Authorization": "Bearer demo-key-for-testing"},
            "expected_status": 200
        },
        {
            "name": "Game Recommendations",
            "method": "POST",
            "url": f"{base_url}/recommend",
            "headers": {
                "Authorization": "Bearer demo-key-for-testing",
                "Content-Type": "application/json"
            },
            "data": {
                "query": "I love The Witcher 3, what similar games would you recommend?",
                "user_id": "test_user_001",
                "max_results": 5,
                "strategy": "hybrid_search"
            },
            "expected_status": 200
        }
    ]
    
    results = []
    
    for test in tests:
        try:
            if test["method"] == "GET":
                response = requests.get(
                    test["url"],
                    headers=test.get("headers", {}),
                    timeout=30
                )
            else:
                response = requests.post(
                    test["url"],
                    headers=test.get("headers", {}),
                    json=test.get("data"),
                    timeout=30
                )
            
            success = response.status_code == test["expected_status"]
            status_icon = "✅" if success else "❌"
            
            print(f"   {status_icon} {test['name']}: {response.status_code}")
            
            if success and test["name"] == "Game Recommendations":
                data = response.json()
                print(f"      📊 Returned {data['total_results']} recommendations")
                print(f"      ⚡ Response time: {data['processing_time_ms']:.1f}ms")
            
            results.append({"test": test["name"], "success": success, "status": response.status_code})
            
        except Exception as e:
            print(f"   ❌ {test['name']}: Error - {e}")
            results.append({"test": test["name"], "success": False, "error": str(e)})
    
    return results

def show_api_documentation():
    """Show API documentation information"""
    
    print("\n📚 API Documentation")
    print("=" * 50)
    print("FastAPI automatically generates interactive documentation:")
    print("   • Swagger UI: http://localhost:8000/docs")
    print("   • ReDoc: http://localhost:8000/redoc")
    print("   • OpenAPI Schema: http://localhost:8000/openapi.json")
    
    print("\n🔑 Authentication")
    print("Use one of these API keys in Authorization header:")
    print("   • Bearer gamematch-api-key-2024")
    print("   • Bearer demo-key-for-testing")
    
    print("\n🎮 Example API Calls")
    print("-" * 20)
    
    examples = [
        {
            "name": "Get Recommendations",
            "method": "POST",
            "url": "/recommend",
            "body": {
                "query": "fantasy RPG games with great story",
                "user_id": "user123",
                "max_results": 10,
                "strategy": "hybrid_search",
                "include_reasoning": True
            }
        },
        {
            "name": "Submit Feedback",
            "method": "POST",
            "url": "/feedback",
            "body": {
                "recommendation_id": "123",
                "feedback_type": "positive",
                "rating": 5,
                "clicked_games": [456, 789]
            }
        },
        {
            "name": "Search Games",
            "method": "GET",
            "url": "/analytics/games/search?q=witcher&limit=10"
        }
    ]
    
    for example in examples:
        print(f"\n{example['name']}:")
        print(f"   {example['method']} {example['url']}")
        if "body" in example:
            print(f"   Body: {json.dumps(example['body'], indent=6)}")

def main():
    """Main function to run the production API"""
    
    print("🎮 GameMatch Production API Launcher")
    print("=" * 60)
    
    # Check dependencies
    deps = check_dependencies()
    
    # Test SQL connection
    sql_ok = test_sql_connection()
    
    # Show API documentation
    show_api_documentation()
    
    print(f"\n🚀 Ready to launch GameMatch API!")
    print("=" * 60)
    
    # Start server (this will block)
    try:
        start_api_server(port=8000, reload=True)
    except KeyboardInterrupt:
        print("\n👋 Shutting down GameMatch API...")
    except Exception as e:
        print(f"\n❌ Server error: {e}")

if __name__ == "__main__":
    main()