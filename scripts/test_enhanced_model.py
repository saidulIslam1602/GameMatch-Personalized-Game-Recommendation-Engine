#!/usr/bin/env python3
"""
Test the enhanced fine-tuned GameMatch model with sophisticated queries
"""

import json
import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from models.openai_finetuning import GameMatchFineTuner

def test_enhanced_capabilities():
    """Test the enhanced capabilities of our fine-tuned model"""
    
    # Load the model ID from results
    results_path = Path(__file__).parent.parent / "results" / "finetuning_results.json"
    
    try:
        with open(results_path, 'r') as f:
            results = json.load(f)
        model_id = results['fine_tuned_model']
        print(f"🤖 Testing model: {model_id}")
    except:
        print("❌ Could not load model ID from results")
        return
    
    # Initialize fine-tuner
    fine_tuner = GameMatchFineTuner()
    
    # Enhanced test queries to check our improvements
    enhanced_tests = [
        # 1. CONTEXTUAL AWARENESS (NEW!)
        {
            "query": "I have 30 minutes before work and want something quick and engaging on PC",
            "type": "⏰ CONTEXTUAL"
        },
        
        # 2. MULTI-TURN CONVERSATION (NEW!)
        {
            "query": "I loved Hades. What else would you recommend?",
            "type": "💬 CONVERSATIONAL"
        },
        
        # 3. PSYCHOLOGICAL PERSONALIZATION (ENHANCED!)
        {
            "query": "I'm an introvert who loves deep stories and exploring beautiful worlds alone",
            "type": "🧠 PSYCHOLOGICAL"
        },
        
        # 4. EXPERT GENRE CURATION (ENHANCED!)
        {
            "query": "What are the most innovative indie puzzle games with unique mechanics?",
            "type": "🎨 EXPERT CURATION"
        },
        
        # 5. COMPARISON & ANALYSIS (NEW!)
        {
            "query": "Compare Elden Ring vs The Witcher 3 - which should I play first?",
            "type": "⚖️ COMPARISON"
        }
    ]
    
    print("🚀 TESTING ENHANCED GAMEMATCH CAPABILITIES")
    print("=" * 60)
    
    for i, test in enumerate(enhanced_tests, 1):
        print(f"\n🎯 TEST {i}: {test['type']}")
        print(f"📝 Query: {test['query']}")
        print("-" * 50)
        
        try:
            response = fine_tuner.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": "You are GameMatch, an AI game recommendation expert. Provide personalized, contextual game recommendations with detailed reasoning."},
                    {"role": "user", "content": test['query']}
                ],
                max_tokens=300,
                temperature=0.7
            )
            
            print(f"🤖 Response:\n{response.choices[0].message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

if __name__ == "__main__":
    test_enhanced_capabilities()