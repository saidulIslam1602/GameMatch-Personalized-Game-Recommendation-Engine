#!/usr/bin/env python3
"""
GameMatch Demo - Show Visual Recommendations
Demonstrates the recommendation system with visual output
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from src.data.dataset_loader import SteamDatasetLoader

def create_visual_demo():
    """Create a visual demonstration of game recommendations"""
    print("🎮 GAMEMATCH VISUAL DEMO")
    print("=" * 50)
    
    # Load data
    print("📂 Loading game data...")
    try:
        loader = SteamDatasetLoader()
        df = loader.load_processed_data()
        print(f"✅ Loaded {len(df)} games")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        print("💡 Make sure you have processed data first:")
        print("   python3 src/data/dataset_loader.py")
        return
    
    # Demo queries
    demo_queries = [
        "action games",
        "strategy",
        "indie",
        "multiplayer",
        "puzzle",
        "RPG",
        "simulation",
        "adventure"
    ]
    
    print("\n🎯 DEMO RECOMMENDATIONS")
    print("=" * 50)
    
    for query in demo_queries:
        print(f"\n🔍 Query: '{query}'")
        print("-" * 30)
        
        # Simple similarity search
        query_lower = query.lower()
        mask = (
            df['Name'].str.lower().str.contains(query_lower, na=False) |
            df['Genres'].str.lower().str.contains(query_lower, na=False) |
            df['Tags'].str.lower().str.contains(query_lower, na=False)
        )
        
        matching_games = df[mask].head(3)
        
        if len(matching_games) == 0:
            print("   No games found")
            continue
        
        for i, (_, game) in enumerate(matching_games.iterrows(), 1):
            rating = game['Review_Score']
            rating_pct = int(rating * 100) if pd.notna(rating) else 0
            price = game['Price']
            price_str = f"${price:.2f}" if pd.notna(price) and price > 0 else "Free"
            
            # Create visual rating bar
            rating_bars = "█" * (rating_pct // 10) + "░" * (10 - rating_pct // 10)
            
            print(f"   {i}. {game['Name']}")
            print(f"      💰 Price: {price_str}")
            print(f"      ⭐ Rating: {rating_pct}% {rating_bars}")
            print(f"      🎮 Genres: {game['Genres']}")
            print(f"      📅 Released: {game['Release date']}")
            
            # Platform indicators
            platforms = []
            if game.get('Windows', False): platforms.append("🪟")
            if game.get('Mac', False): platforms.append("🍎")
            if game.get('Linux', False): platforms.append("🐧")
            if platforms:
                print(f"      💻 Platforms: {' '.join(platforms)}")
            
            print()

def create_recommendation_html():
    """Create an HTML file with visual recommendations"""
    print("\n🌐 Creating visual HTML demo...")
    
    # Load data
    try:
        loader = SteamDatasetLoader()
        df = loader.load_processed_data()
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None
    
    # Sample recommendations
    sample_games = df.sample(12)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GameMatch - Visual Recommendations</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                margin: 0;
                padding: 20px;
                min-height: 100vh;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding: 20px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                color: white;
                border-radius: 15px;
            }}
            .games-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin-top: 30px;
            }}
            .game-card {{
                background: white;
                border-radius: 15px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                overflow: hidden;
                transition: transform 0.3s ease;
            }}
            .game-card:hover {{
                transform: translateY(-5px);
            }}
            .game-image {{
                width: 100%;
                height: 200px;
                object-fit: cover;
            }}
            .game-info {{
                padding: 20px;
            }}
            .game-title {{
                font-size: 18px;
                font-weight: bold;
                margin-bottom: 10px;
                color: #333;
            }}
            .game-genres {{
                color: #667eea;
                font-size: 14px;
                margin-bottom: 10px;
            }}
            .game-price {{
                font-size: 20px;
                font-weight: bold;
                color: #28a745;
                margin-bottom: 10px;
            }}
            .rating {{
                display: flex;
                align-items: center;
                margin-bottom: 10px;
            }}
            .rating-bar {{
                width: 100px;
                height: 10px;
                background: #e9ecef;
                border-radius: 5px;
                overflow: hidden;
                margin-left: 10px;
            }}
            .rating-fill {{
                height: 100%;
                background: linear-gradient(45deg, #ffd700, #ffed4e);
                transition: width 0.3s ease;
            }}
            .platforms {{
                margin-top: 10px;
            }}
            .platform {{
                display: inline-block;
                background: #e9ecef;
                color: #495057;
                padding: 3px 8px;
                border-radius: 10px;
                font-size: 11px;
                margin-right: 5px;
            }}
            .platform.active {{
                background: #28a745;
                color: white;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎮 GameMatch - AI Game Recommendations</h1>
                <p>Discover your next favorite game with intelligent recommendations</p>
            </div>
            
            <div class="games-grid">
    """
    
    for _, game in sample_games.iterrows():
        rating = game['Review_Score']
        rating_pct = int(rating * 100) if pd.notna(rating) else 0
        price = game['Price']
        price_str = f"${price:.2f}" if pd.notna(price) and price > 0 else "Free"
        
        # Platform indicators
        platforms = []
        if game.get('Windows', False): platforms.append('<span class="platform active">Windows</span>')
        if game.get('Mac', False): platforms.append('<span class="platform active">Mac</span>')
        if game.get('Linux', False): platforms.append('<span class="platform active">Linux</span>')
        
        html_content += f"""
                <div class="game-card">
                    <img src="{game['Header image'] or '/static/placeholder.jpg'}" 
                         class="game-image" alt="{game['Name']}">
                    <div class="game-info">
                        <div class="game-title">{game['Name']}</div>
                        <div class="game-genres">{game['Genres'] or 'No genres'}</div>
                        <div class="game-price">{price_str}</div>
                        <div class="rating">
                            Rating: {rating_pct}%
                            <div class="rating-bar">
                                <div class="rating-fill" style="width: {rating_pct}%"></div>
                            </div>
                        </div>
                        <div class="platforms">
                            {''.join(platforms)}
                        </div>
                    </div>
                </div>
        """
    
    html_content += """
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    html_file = project_root / "gamematch_demo.html"
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Visual demo saved to: {html_file}")
    print(f"🌐 Open in browser: file://{html_file.absolute()}")
    
    return html_file

def main():
    """Main demo function"""
    try:
        # Create console demo
        create_visual_demo()
        
        # Create HTML demo
        html_file = create_recommendation_html()
        
        print("\n🎉 DEMO COMPLETE!")
        print("=" * 50)
        print("📊 Console demo: Shows text-based recommendations")
        print(f"🌐 Visual demo: {html_file}")
        print("\n💡 To see the full interactive dashboard:")
        print("   python3 scripts/run_dashboard.py")
        
    except Exception as e:
        print(f"❌ Error creating demo: {e}")
        print("💡 Make sure you have processed data first:")
        print("   python3 src/data/dataset_loader.py")

if __name__ == "__main__":
    main()