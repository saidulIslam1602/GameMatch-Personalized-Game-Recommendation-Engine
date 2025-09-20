#!/usr/bin/env python3
"""
GameMatch Fine-tuning Status Checker
Monitor your OpenAI fine-tuning job progress
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.openai_finetuning import GameMatchFineTuner
import time

def main():
    """Check fine-tuning job status"""
    print("🔍 GAMEMATCH FINE-TUNING STATUS CHECKER")
    print("=" * 50)
    
    # Load results to get job ID
    results_file = Path("results") / "finetuning_results.json"
    
    if not results_file.exists():
        print("❌ No fine-tuning results found!")
        print("Make sure you've run the fine-tuning pipeline first.")
        return
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    job_id = results.get('job_id')
    if not job_id:
        print("❌ No job ID found in results!")
        return
    
    print(f"📋 Checking job: {job_id}")
    
    # Initialize fine-tuner
    finetuner = GameMatchFineTuner()
    
    if not finetuner.api_key:
        print("❌ OpenAI API key not found!")
        return
    
    try:
        # Check status
        status = finetuner.check_job_status(job_id)
        
        if isinstance(status, str) and status.startswith("ft:"):
            print(f"🎉 FINE-TUNING COMPLETED!")
            print(f"✅ Model ID: {status}")
            print("\n🧪 Testing the model...")
            
            # Test the model
            finetuner.test_fine_tuned_model(status)
            
            # Save model ID
            results['fine_tuned_model'] = status
            results['status'] = 'completed'
            
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"💾 Results updated: {results_file}")
        
        elif status == "succeeded":
            print("🎉 Fine-tuning succeeded but model ID not retrieved yet.")
            print("Please run this script again in a few minutes.")
        
        elif status in ["validating_files", "queued", "running"]:
            print(f"⏳ Fine-tuning in progress: {status}")
            print("Fine-tuning typically takes 10-30 minutes.")
            print("Run this script again in a few minutes to check progress.")
        
        elif status == "failed":
            print("❌ Fine-tuning failed! Check the OpenAI dashboard for details.")
        
        else:
            print(f"📊 Current status: {status}")
            
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    main()