#!/usr/bin/env python3
"""
GameMatch Fine-tuning Runner
Easy-to-use script to run the OpenAI fine-tuning pipeline
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.openai_finetuning import GameMatchFineTuner
import logging
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run the GameMatch fine-tuning pipeline"""
    print("🎮 GAMEMATCH OPENAI FINE-TUNING PIPELINE")
    print("=" * 50)
    
    try:
        # Initialize fine-tuner
        finetuner = GameMatchFineTuner()
        
        # Check if we have an API key
        if not finetuner.api_key:
            print("❌ OpenAI API key not found!")
            print("Make sure the API key is stored in config/openai_key.txt")
            return
        
        print(f"✅ OpenAI API key loaded (ends with: ...{finetuner.api_key[-10:]})")
        
        # Run the complete pipeline
        result = finetuner.run_complete_pipeline()
        
        # Save results
        results_file = Path("results") / "finetuning_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"\n📊 RESULTS SAVED TO: {results_file}")
        print("\n🎯 NEXT STEPS:")
        print("1. Monitor the fine-tuning job progress")
        print("2. Once complete, test the fine-tuned model")
        print("3. Deploy for production use")
        
        print(f"\n📋 JOB DETAILS:")
        print(f"   • Job ID: {result['job_id']}")
        print(f"   • Training File: {result['training_file']}")
        print(f"   • Status: {result['status']}")
        
    except Exception as e:
        logger.error(f"❌ Fine-tuning pipeline failed: {e}")
        raise

if __name__ == "__main__":
    main()