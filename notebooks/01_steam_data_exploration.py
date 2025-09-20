"""
GameMatch: Steam Dataset Exploration
Comprehensive exploratory data analysis of the Steam games dataset
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import ast
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the processed Steam games dataset"""
    print("📂 Loading processed Steam games dataset...")
    
    # Load processed data
    processed_file = Path("data/processed/steam_games_processed.parquet")
    analysis_file = Path("data/processed/steam_dataset_analysis.json")
    
    if not processed_file.exists():
        raise FileNotFoundError("Processed dataset not found. Run dataset_loader.py first.")
    
    df = pd.read_parquet(processed_file)
    
    with open(analysis_file, 'r') as f:
        analysis = json.load(f)
    
    print(f"✅ Loaded {len(df):,} games")
    return df, analysis

def analyze_game_distribution(df):
    """Analyze the distribution of games across different dimensions"""
    print("\n🎮 GAME DISTRIBUTION ANALYSIS")
    print("=" * 50)
    
    # Release year distribution
    if 'Release_Year' in df.columns:
        print(f"📅 Release Year Range: {df['Release_Year'].min():.0f} - {df['Release_Year'].max():.0f}")
        
        # Games by decade
        decade_counts = df.groupby(df['Release_Year'] // 10 * 10).size()
        print("\n🗓️  Games by Decade:")
        for decade, count in decade_counts.tail(6).items():  # Last 6 decades
            print(f"   {decade}s: {count:,} games")
    
    # Price distribution
    if 'Price' in df.columns:
        free_games = (df['Price'] == 0).sum()
        paid_games = (df['Price'] > 0).sum()
        
        print(f"\n💰 Price Distribution:")
        print(f"   Free games: {free_games:,} ({free_games/len(df)*100:.1f}%)")
        print(f"   Paid games: {paid_games:,} ({paid_games/len(df)*100:.1f}%)")
        
        if paid_games > 0:
            paid_df = df[df['Price'] > 0]
            print(f"   Average price: ${paid_df['Price'].mean():.2f}")
            print(f"   Median price: ${paid_df['Price'].median():.2f}")
            print(f"   Price range: ${paid_df['Price'].min():.2f} - ${paid_df['Price'].max():.2f}")

def analyze_ratings_and_reviews(df):
    """Analyze ratings and review patterns"""
    print("\n⭐ RATINGS & REVIEWS ANALYSIS")
    print("=" * 50)
    
    if 'Total_Reviews' in df.columns:
        reviewed_games = df[df['Total_Reviews'] > 0]
        print(f"📝 Games with reviews: {len(reviewed_games):,} ({len(reviewed_games)/len(df)*100:.1f}%)")
        
        if len(reviewed_games) > 0:
            print(f"   Average reviews per game: {reviewed_games['Total_Reviews'].mean():.0f}")
            print(f"   Median reviews per game: {reviewed_games['Total_Reviews'].median():.0f}")
            
            # Review score distribution
            if 'Review_Score' in reviewed_games.columns:
                avg_score = reviewed_games['Review_Score'].mean()
                print(f"   Average review score: {avg_score:.3f} ({avg_score*100:.1f}%)")
                
                # Score categories
                excellent = (reviewed_games['Review_Score'] >= 0.9).sum()
                good = ((reviewed_games['Review_Score'] >= 0.7) & (reviewed_games['Review_Score'] < 0.9)).sum()
                mixed = ((reviewed_games['Review_Score'] >= 0.4) & (reviewed_games['Review_Score'] < 0.7)).sum()
                poor = (reviewed_games['Review_Score'] < 0.4).sum()
                
                print(f"\n📊 Review Score Categories:")
                print(f"   Excellent (90%+): {excellent:,} games")
                print(f"   Good (70-89%): {good:,} games") 
                print(f"   Mixed (40-69%): {mixed:,} games")
                print(f"   Poor (<40%): {poor:,} games")

def analyze_genres_and_categories(df):
    """Analyze genres, categories, and tags"""
    print("\n🏷️  GENRES & CATEGORIES ANALYSIS")
    print("=" * 50)
    
    # Analyze genres
    if 'Genres_List' in df.columns:
        all_genres = []
        for genres_list in df['Genres_List'].dropna():
            if isinstance(genres_list, list):
                all_genres.extend(genres_list)
        
        if all_genres:
            genre_counts = pd.Series(all_genres).value_counts()
            print(f"🎯 Top 10 Genres:")
            for genre, count in genre_counts.head(10).items():
                print(f"   {genre}: {count:,} games")
    
    # Analyze categories
    if 'Categories_List' in df.columns:
        all_categories = []
        for cat_list in df['Categories_List'].dropna():
            if isinstance(cat_list, list):
                all_categories.extend(cat_list)
        
        if all_categories:
            cat_counts = pd.Series(all_categories).value_counts()
            print(f"\n📂 Top 10 Categories:")
            for category, count in cat_counts.head(10).items():
                print(f"   {category}: {count:,} games")

def analyze_platforms(df):
    """Analyze platform distribution"""
    print("\n💻 PLATFORM ANALYSIS") 
    print("=" * 50)
    
    platform_cols = ['Windows', 'Mac', 'Linux']
    available_platforms = [col for col in platform_cols if col in df.columns]
    
    if available_platforms:
        for platform in available_platforms:
            count = df[platform].sum()
            percentage = count / len(df) * 100
            print(f"   {platform}: {count:,} games ({percentage:.1f}%)")
        
        # Multi-platform games
        if 'Platform_Count' in df.columns:
            platform_dist = df['Platform_Count'].value_counts().sort_index()
            print(f"\n🔗 Multi-platform Distribution:")
            for count, games in platform_dist.items():
                if count == 1:
                    label = "Single platform"
                elif count == 2:
                    label = "Two platforms" 
                else:
                    label = "All platforms"
                print(f"   {label}: {games:,} games")

def analyze_developers_and_publishers(df):
    """Analyze developer and publisher patterns"""
    print("\n👥 DEVELOPERS & PUBLISHERS ANALYSIS")
    print("=" * 50)
    
    # Top developers
    if 'Developers' in df.columns:
        dev_counts = df['Developers'].value_counts()
        print(f"🔧 Top 10 Developers by Game Count:")
        for dev, count in dev_counts.head(10).items():
            if pd.notna(dev) and dev != 'nan':
                print(f"   {dev}: {count:,} games")
    
    # Top publishers  
    if 'Publishers' in df.columns:
        pub_counts = df['Publishers'].value_counts()
        print(f"\n📢 Top 10 Publishers by Game Count:")
        for pub, count in pub_counts.head(10).items():
            if pd.notna(pub) and pub != 'nan':
                print(f"   {pub}: {count:,} games")

def create_recommendation_insights(df):
    """Generate insights for recommendation system design"""
    print("\n🎯 ENHANCED RECOMMENDATION SYSTEM INSIGHTS")
    print("=" * 50)
    
    # Enhanced data quality metrics
    quality_metrics = {}
    
    if 'Data_Quality_Score' in df.columns:
        avg_quality = df['Data_Quality_Score'].mean()
        high_quality = (df['Data_Quality_Score'] >= 7).sum()  # Higher threshold for enhanced scoring
        quality_metrics['average_quality_score'] = avg_quality
        quality_metrics['high_quality_games'] = high_quality
        print(f"📊 Enhanced quality score: {avg_quality:.2f}/10 (was /4)")
        print(f"🏆 High quality games (score ≥7): {high_quality:,} ({high_quality/len(df)*100:.1f}%)")
    
    # Content richness analysis
    if 'Content_Richness' in df.columns:
        avg_richness = df['Content_Richness'].mean()
        rich_content = (df['Content_Richness'] >= 0.7).sum()
        print(f"📝 Average content richness: {avg_richness:.3f}")
        print(f"🎨 Rich content games (≥0.7): {rich_content:,} ({rich_content/len(df)*100:.1f}%)")
    
    # Commercial viability analysis
    if 'Commercial_Viability' in df.columns:
        avg_viability = df['Commercial_Viability'].mean()
        viable_games = (df['Commercial_Viability'] >= 0.6).sum()
        print(f"💼 Average commercial viability: {avg_viability:.3f}")
        print(f"💰 Viable games (≥0.6): {viable_games:,} ({viable_games/len(df)*100:.1f}%)")
    
    # Enhanced content features
    feature_availability = {}
    enhanced_features = ['Genres_List', 'Categories_List', 'Tags_List', 'About the game', 
                        'Description_Sentiment', 'Wilson_Score', 'Price_Category']
    
    print(f"\n✨ Enhanced Feature Availability:")
    for feature in enhanced_features:
        if feature in df.columns:
            if feature == 'Description_Sentiment':
                available = (df[feature] != 0).sum()  # Non-neutral sentiment
                print(f"   {feature}: {available:,} games ({available/len(df)*100:.1f}%) have sentiment")
            elif feature == 'Wilson_Score':
                available = (df[feature] > 0).sum()
                print(f"   {feature}: {available:,} games ({available/len(df)*100:.1f}%) have reliable ratings")
            elif feature == 'Price_Category':
                if hasattr(df[feature], 'value_counts'):
                    print(f"   {feature}: {df[feature].notna().sum():,} categorized")
            else:
                available = df[feature].notna().sum()
                feature_availability[feature] = available
                print(f"   {feature}: {available:,} games ({available/len(df)*100:.1f}%)")
    
    # Advanced rating analysis
    print(f"\n📈 Advanced Rating Metrics:")
    if 'Wilson_Score' in df.columns and 'Review_Score' in df.columns:
        reliable_ratings = (df['Wilson_Score'] > 0.5).sum()
        print(f"   Reliable ratings (Wilson >0.5): {reliable_ratings:,} games")
        
        if 'Controversy_Score' in df.columns:
            controversial = (df['Controversy_Score'] > 0.3).sum()
            print(f"   Controversial games (score >0.3): {controversial:,} games")
    
    # Enhanced collaborative filtering
    if 'Total_Reviews' in df.columns:
        games_with_reviews = (df['Total_Reviews'] > 0).sum()
        avg_reviews = df[df['Total_Reviews'] > 0]['Total_Reviews'].mean()
        high_engagement = (df['Total_Reviews'] > 100).sum()
        print(f"\n🤝 Enhanced Collaborative Filtering:")
        print(f"   Games with reviews: {games_with_reviews:,} ({games_with_reviews/len(df)*100:.1f}%)")
        print(f"   High engagement (>100 reviews): {high_engagement:,} games")
        print(f"   Average reviews per game: {avg_reviews:.0f}")
    
    return quality_metrics, feature_availability

def generate_dataset_summary(df, analysis):
    """Generate a comprehensive dataset summary"""
    print("\n📋 DATASET SUMMARY FOR GAMEMATCH")
    print("=" * 60)
    
    print(f"🎮 Total Games: {len(df):,}")
    print(f"💾 Dataset Size: {analysis.get('basic_info', {}).get('memory_usage', 'N/A')}")
    print(f"🗂️  Features: {len(df.columns)} columns")
    
    # Key statistics
    if 'Release_Year' in df.columns:
        year_range = f"{df['Release_Year'].min():.0f}-{df['Release_Year'].max():.0f}"
        print(f"📅 Time Range: {year_range}")
    
    if 'Total_Reviews' in df.columns:
        total_reviews = df['Total_Reviews'].sum()
        print(f"📝 Total User Reviews: {total_reviews:,.0f}")
    
    # Recommendation system readiness
    print(f"\n🚀 RECOMMENDATION SYSTEM READINESS:")
    
    # Content-based features
    content_features = ['Genres', 'Categories', 'Tags', 'About the game']
    available_content = sum(1 for feat in content_features if feat in df.columns)
    print(f"   ✅ Content features: {available_content}/{len(content_features)}")
    
    # Collaborative features
    collab_ready = 'Total_Reviews' in df.columns and df['Total_Reviews'].sum() > 0
    print(f"   ✅ Collaborative filtering: {'Ready' if collab_ready else 'Limited'}")
    
    # Hierarchical classification
    hierarchical_ready = 'Genres' in df.columns or 'Categories' in df.columns
    print(f"   ✅ Hierarchical classification: {'Ready' if hierarchical_ready else 'Limited'}")
    
    print(f"\n🎯 Ready for GameMatch development! 🚀")

def main():
    """Main exploration function"""
    print("🎮 GAMEMATCH: STEAM DATASET EXPLORATION")
    print("=" * 60)
    
    # Load data
    df, analysis = load_data()
    
    # Run all analyses
    analyze_game_distribution(df)
    analyze_ratings_and_reviews(df)
    analyze_genres_and_categories(df) 
    analyze_platforms(df)
    analyze_developers_and_publishers(df)
    quality_metrics, feature_availability = create_recommendation_insights(df)
    
    # Generate final summary
    generate_dataset_summary(df, analysis)
    
    return df, analysis, quality_metrics, feature_availability

if __name__ == "__main__":
    df, analysis, quality_metrics, feature_availability = main()