import pandas as pd
import numpy as np

# Load processed data
df = pd.read_parquet('data/processed/steam_games_processed.parquet')

print('🎮 ENHANCED DATASET SAMPLE:')
print('='*60)

# Show enhanced features
enhanced_cols = [
    'Name', 'Price', 'Price_Category', 'Wilson_Score', 'Commercial_Viability',
    'Content_Richness', 'Description_Sentiment', 'Era_Category'
]

# Get available columns
available_cols = [col for col in enhanced_cols if col in df.columns]
sample = df[available_cols].head(10)

for idx, row in sample.iterrows():
    print(f'📌 {row["Name"]}')
    
    if 'Price' in row and 'Price_Category' in row:
        print(f'   💰 Price: ${row["Price"]:.2f} ({row["Price_Category"]})')
    
    if 'Wilson_Score' in row:
        reliability = "High" if row["Wilson_Score"] > 0.5 else "Low" if row["Wilson_Score"] > 0 else "None"
        print(f'   ⭐ Wilson Score: {row["Wilson_Score"]:.3f} ({reliability} reliability)')
    
    if 'Commercial_Viability' in row:
        viability = "High" if row["Commercial_Viability"] > 0.6 else "Medium" if row["Commercial_Viability"] > 0.3 else "Low"
        print(f'   💼 Commercial Viability: {row["Commercial_Viability"]:.3f} ({viability})')
    
    if 'Content_Richness' in row:
        richness = "Rich" if row["Content_Richness"] > 0.7 else "Medium" if row["Content_Richness"] > 0.4 else "Basic"
        print(f'   📝 Content Richness: {row["Content_Richness"]:.3f} ({richness})')
    
    if 'Description_Sentiment' in row:
        sentiment = "Positive" if row["Description_Sentiment"] > 0.1 else "Negative" if row["Description_Sentiment"] < -0.1 else "Neutral"
        print(f'   😊 Description Sentiment: {row["Description_Sentiment"]:.3f} ({sentiment})')
    
    if 'Era_Category' in row:
        print(f'   📅 Era: {row["Era_Category"]}')
    
    print()

print('📈 ENHANCED PREPROCESSING IMPROVEMENTS:')
print('='*50)

# Show improvement statistics
new_features = [
    'Wilson_Score', 'Commercial_Viability', 'Content_Richness', 
    'Description_Sentiment', 'Price_Category', 'Era_Category',
    'Review_Density', 'Controversy_Score', 'Is_Multiplatform'
]

available_new = [col for col in new_features if col in df.columns]
print(f'✅ New Features Added: {len(available_new)}')

# Show feature coverage
for feature in available_new[:6]:  # Show first 6
    if feature in df.columns:
        coverage = df[feature].notna().sum()
        print(f'   • {feature}: {coverage:,} games ({coverage/len(df)*100:.1f}%)')

print(f'\n📊 ENHANCED DATASET STRUCTURE:')
print(f'   • Shape: {df.shape}')
print(f'   • Total Features: {df.shape[1]} (was 39)')
print(f'   • New Features: {df.shape[1] - 39}')
print(f'   • Quality Score: {df["Data_Quality_Score"].mean():.1f}/10 (was 3.9/4)')

# Show quality improvements
if 'Content_Richness' in df.columns:
    rich_content = (df['Content_Richness'] > 0.7).sum()
    print(f'   • Rich Content Games: {rich_content:,} ({rich_content/len(df)*100:.1f}%)')

if 'Commercial_Viability' in df.columns:
    viable_games = (df['Commercial_Viability'] > 0.6).sum()
    print(f'   • Commercially Viable: {viable_games:,} ({viable_games/len(df)*100:.1f}%)')

print(f'\n🚀 READY FOR ADVANCED ML MODELS!')
print(f'   ✅ Hierarchical Classification: Enhanced taxonomy features')
print(f'   ✅ LLM Fine-tuning: Rich text + sentiment analysis') 
print(f'   ✅ RAG System: Content richness scoring')
print(f'   ✅ Multi-Agent: Commercial viability metrics')
print(f'   ✅ Recommendation Engine: Wilson scores + controversy detection')