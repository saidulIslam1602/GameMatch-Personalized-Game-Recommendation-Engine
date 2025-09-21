#!/usr/bin/env python3
"""
Simple GameMatch Visual Demo
Shows game recommendations with visual elements
"""

import os
import sys
import pandas as pd
import json
from pathlib import Path

def load_game_data():
    """Load processed game data"""
    data_file = Path("data/processed/steam_games_processed.parquet")
    
    if not data_file.exists():
        print("❌ Processed data not found. Please run data preprocessing first:")
        print("   python3 src/data/dataset_loader.py")
        return None
    
    try:
        df = pd.read_parquet(data_file)
        print(f"✅ Loaded {len(df)} games")
        return df
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

def create_visual_recommendations(df, query, num_games=5):
    """Create visual recommendations for a query"""
    print(f"\n🔍 Searching for: '{query}'")
    print("=" * 50)
    
    # Simple text search
    query_lower = query.lower()
    mask = (
        df['Name'].str.lower().str.contains(query_lower, na=False) |
        df['Genres'].str.lower().str.contains(query_lower, na=False) |
        df['Tags'].str.lower().str.contains(query_lower, na=False)
    )
    
    matching_games = df[mask].head(num_games)
    
    if len(matching_games) == 0:
        print("   No games found for this query")
        return
    
    for i, (_, game) in enumerate(matching_games.iterrows(), 1):
        # Game info
        name = game['Name']
        rating = game['Review_Score']
        rating_pct = int(rating * 100) if pd.notna(rating) else 0
        price = game['Price']
        price_str = f"${price:.2f}" if pd.notna(price) and price > 0 else "Free"
        genres = game['Genres'] or "No genres"
        release_date = game['Release date']
        
        # Create visual elements
        rating_bars = "█" * (rating_pct // 10) + "░" * (10 - rating_pct // 10)
        
        # Platform indicators
        platforms = []
        if game.get('Windows', False): platforms.append("🪟")
        if game.get('Mac', False): platforms.append("🍎")
        if game.get('Linux', False): platforms.append("🐧")
        
        # Print game card
        print(f"\n🎮 GAME #{i}")
        print("┌" + "─" * 48 + "┐")
        print(f"│ {name[:46]:<46} │")
        print("├" + "─" * 48 + "┤")
        print(f"│ 💰 Price: {price_str:<20} ⭐ {rating_pct}% {rating_bars[:15]:<15} │")
        print(f"│ 🎭 Genres: {genres[:35]:<35} │")
        print(f"│ 📅 Released: {str(release_date)[:25]:<25} │")
        if platforms:
            print(f"│ 💻 Platforms: {' '.join(platforms):<30} │")
        print("└" + "─" * 48 + "┘")

def create_html_demo(df):
    """Create an HTML file with visual recommendations"""
    print("\n🌐 Creating HTML visual demo...")
    
    # Sample games for demo
    sample_games = df.sample(12)
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GameMatch - AI Game Recommendations</title>
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
            border: 2px solid #e9ecef;
        }}
        .game-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.15);
        }}
        .game-image {{
            width: 100%;
            height: 200px;
            object-fit: cover;
            background: linear-gradient(45deg, #f8f9fa, #e9ecef);
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
        .stats {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🎮 GameMatch - AI Game Recommendation Engine</h1>
            <p>Discover your next favorite game with intelligent recommendations</p>
        </div>
        
        <div class="stats">
            <h3>📊 Dataset Statistics</h3>
            <p><strong>Total Games:</strong> {len(df):,}</p>
            <p><strong>Genres Available:</strong> {len(df['Genres'].dropna().unique())}</p>
            <p><strong>Price Range:</strong> ${df['Price'].min():.2f} - ${df['Price'].max():.2f}</p>
        </div>
        
        <h3>🎯 Featured Game Recommendations</h3>
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
                <img src="{game['Header image'] or 'data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzAwIiBoZWlnaHQ9IjIwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSIjZjhmOWZhIi8+PHRleHQgeD0iNTAlIiB5PSI1MCUiIGZvbnQtZmFtaWx5PSJBcmlhbCIgZm9udC1zaXplPSIxNCIgZmlsbD0iIzk5OSIgdGV4dC1hbmNob3I9Im1pZGRsZSIgZHk9Ii4zZW0iPkdhbWUgSW1hZ2U8L3RleHQ+PC9zdmc+'}" 
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
    html_file = Path("gamematch_visual_demo.html")
    with open(html_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"✅ Visual demo saved to: {html_file.absolute()}")
    print(f"🌐 Open in browser: file://{html_file.absolute()}")
    
    return html_file

def main():
    """Main demo function"""
    print("🎮 GAMEMATCH VISUAL DEMO")
    print("=" * 50)
    
    # Load data
    df = load_game_data()
    if df is None:
        return
    
    # Demo queries
    demo_queries = [
        "action",
        "strategy", 
        "indie",
        "multiplayer",
        "puzzle",
        "RPG"
    ]
    
    print("\n🎯 INTERACTIVE RECOMMENDATIONS")
    print("=" * 50)
    
    for query in demo_queries:
        create_visual_recommendations(df, query, 3)
    
    # Create HTML demo
    html_file = create_html_demo(df)
    
    print("\n🎉 DEMO COMPLETE!")
    print("=" * 50)
    print("📊 Console demo: Shows text-based recommendations above")
    print(f"🌐 Visual demo: {html_file}")
    print("\n💡 To see the full interactive dashboard:")
    print("   python3 scripts/run_dashboard.py")
    print("\n🔍 Try these search terms in the dashboard:")
    for query in demo_queries:
        print(f"   • '{query}'")

if __name__ == "__main__":
    main()