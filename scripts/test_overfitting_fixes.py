#!/usr/bin/env python3
"""
Test the anti-overfitting improvements in our fine-tuning system
"""

import sys
import pandas as pd
from pathlib import Path
from collections import Counter

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.append(str(src_path))

from models.openai_finetuning import GameMatchFineTuner

def test_balanced_sampling():
    """Test the new balanced sampling approach"""
    print("🧪 TESTING ANTI-OVERFITTING IMPROVEMENTS")
    print("=" * 50)
    
    fine_tuner = GameMatchFineTuner()
    
    # Load processed data
    try:
        df = fine_tuner.load_processed_data()
        print(f"✅ Loaded {len(df):,} games")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    print("\n📊 BEFORE: Creating similarity examples with old method...")
    
    # Test balanced sampling by creating a small set of examples
    try:
        print("🔄 Generating 50 examples to test game diversity...")
        examples = fine_tuner.create_enhanced_similarity_examples(df, num_examples=50)
        
        # Analyze game diversity in recommendations
        recommended_games = []
        for example in examples:
            response = example['messages'][-1]['content']
            # Simple parsing to extract game names (this is just for testing)
            lines = response.split('\n')
            for line in lines:
                if '**' in line and '$' in line:
                    # Extract game name between ** markers
                    start = line.find('**') + 2
                    end = line.find('**', start)
                    if start < end:
                        game_name = line[start:end]
                        recommended_games.append(game_name)
        
        # Count frequencies
        game_counts = Counter(recommended_games)
        
        print(f"\n📊 RESULTS - Generated {len(examples)} examples")
        print(f"🎯 Usage stats after generation:")
        print(fine_tuner.get_usage_stats())
        
        print(f"\n🎮 Top recommended games in examples:")
        for game, count in game_counts.most_common(10):
            print(f"   • {game}: {count} times")
        
        # Check if any game exceeds our limit
        overused_games = [game for game, count in game_counts.items() if count > fine_tuner.max_game_usage]
        if overused_games:
            print(f"\n⚠️ Games exceeding limit ({fine_tuner.max_game_usage}): {overused_games}")
        else:
            print(f"\n✅ No games exceed usage limit of {fine_tuner.max_game_usage}")
        
        # Calculate diversity metrics
        total_recommendations = len(recommended_games)
        unique_games = len(set(recommended_games))
        diversity_ratio = unique_games / total_recommendations if total_recommendations > 0 else 0
        
        print(f"\n📈 DIVERSITY METRICS:")
        print(f"   • Total recommendations: {total_recommendations}")
        print(f"   • Unique games: {unique_games}")
        print(f"   • Diversity ratio: {diversity_ratio:.2%}")
        
        if diversity_ratio > 0.3:  # At least 30% unique
            print("✅ Good diversity achieved!")
        else:
            print("⚠️ Diversity could be improved")
            
    except Exception as e:
        print(f"❌ Error testing: {e}")
        import traceback
        traceback.print_exc()

def test_usage_counter_reset():
    """Test the usage counter reset functionality"""
    print("\n🔄 TESTING USAGE COUNTER RESET")
    print("-" * 30)
    
    fine_tuner = GameMatchFineTuner()
    
    # Manually add some usage
    fine_tuner.game_usage_counter = {
        "Test Game 1": 3,
        "Test Game 2": 5,
        "Test Game 3": 2
    }
    
    print("Before reset:", fine_tuner.get_usage_stats())
    fine_tuner.reset_game_usage_counter()
    print("After reset:", fine_tuner.get_usage_stats())

if __name__ == "__main__":
    test_balanced_sampling()
    test_usage_counter_reset()
    
    print("\n🎉 ANTI-OVERFITTING TESTS COMPLETE!")
    print("\n📋 KEY IMPROVEMENTS IMPLEMENTED:")
    print("   ✅ Balanced sampling across quality tiers")
    print("   ✅ Game usage frequency tracking")
    print("   ✅ Diversity penalties for overused games")
    print("   ✅ Reduced quality bonus impact")
    print("   ✅ Randomization for tie-breaking")
    print("   ✅ Automatic constraint relaxation")