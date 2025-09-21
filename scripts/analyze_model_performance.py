#!/usr/bin/env python3
"""
Analyze the fine-tuned model performance and test different prompting strategies
"""

import json
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from models.openai_finetuning import GameMatchFineTuner

def test_prompt_variations():
    """Test different prompting strategies to improve recommendations"""
    
    # Load model ID
    results_path = Path(__file__).parent.parent / "results" / "finetuning_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    model_id = results['fine_tuned_model']
    
    fine_tuner = GameMatchFineTuner()
    
    # Test query
    test_query = "I loved The Witcher 3. What else would you recommend?"
    
    # Different prompting strategies
    prompt_strategies = [
        {
            "name": "🎯 SPECIFIC SYSTEM PROMPT",
            "system": """You are GameMatch, a game recommendation expert. When recommending games:
1. Analyze the specific game mentioned to understand why they liked it
2. Recommend games with SIMILAR core mechanics, themes, or genres
3. Avoid repeating the same games constantly
4. Provide diverse recommendations from your training data
5. Include detailed reasoning about similarities"""
        },
        {
            "name": "🎮 GENRE-FOCUSED PROMPT", 
            "system": """You are GameMatch. The user loved The Witcher 3 (open-world RPG with rich story, choices, fantasy setting, exploration). Recommend games that share these core elements. Prioritize variety and explain similarities."""
        },
        {
            "name": "🔥 TEMPERATURE VARIATION",
            "system": "You are GameMatch, an AI game recommendation expert. Provide personalized, diverse game recommendations with detailed reasoning.",
            "temperature": 1.2
        }
    ]
    
    print("🔍 ANALYZING MODEL PERFORMANCE WITH DIFFERENT PROMPTS")
    print("=" * 60)
    
    for i, strategy in enumerate(prompt_strategies, 1):
        print(f"\n🧪 STRATEGY {i}: {strategy['name']}")
        print(f"📝 Query: {test_query}")
        print("-" * 50)
        
        try:
            response = fine_tuner.client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": strategy['system']},
                    {"role": "user", "content": test_query}
                ],
                max_tokens=400,
                temperature=strategy.get('temperature', 0.7)
            )
            
            print(f"🤖 Response:\n{response.choices[0].message.content}")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print("-" * 50)

def analyze_training_data_bias():
    """Analyze potential bias in our training data"""
    
    print("\n📊 TRAINING DATA ANALYSIS")
    print("=" * 40)
    
    # Load training data
    training_file = Path(__file__).parent.parent / "src" / "models" / "gamematch_enhanced_training_data.jsonl"
    
    if not training_file.exists():
        print("❌ Training file not found")
        return
    
    games_mentioned = {}
    
    try:
        with open(training_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                response = data['messages'][-1]['content']
                
                # Simple analysis - count mentions of specific games
                if "Slay the Princess" in response:
                    games_mentioned["Slay the Princess"] = games_mentioned.get("Slay the Princess", 0) + 1
                if "Love Is All Around" in response:
                    games_mentioned["Love Is All Around"] = games_mentioned.get("Love Is All Around", 0) + 1
                if "BOOK OF HOURS" in response:
                    games_mentioned["BOOK OF HOURS"] = games_mentioned.get("BOOK OF HOURS", 0) + 1
    
        print("🎮 Most mentioned games in training data:")
        for game, count in sorted(games_mentioned.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {game}: {count} times")
            
    except Exception as e:
        print(f"❌ Error analyzing training data: {e}")

if __name__ == "__main__":
    test_prompt_variations()
    analyze_training_data_bias()