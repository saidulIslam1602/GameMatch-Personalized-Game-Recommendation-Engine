"""
GameMatch Dataset Loader
Handles downloading and preprocessing of gaming datasets for the GameMatch project
"""

import pandas as pd
import numpy as np
from datasets import load_dataset
from pathlib import Path
import json
import ast
import re
from typing import Dict, List, Tuple, Optional
import logging
from collections import Counter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameMatchDataLoader:
    """Load and preprocess gaming datasets for the GameMatch project"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_steam_games(self, save_raw=True) -> pd.DataFrame:
        """Load Steam games dataset from Hugging Face"""
        logger.info("🎮 Loading Steam Games Dataset from Hugging Face...")
        
        try:
            # Load dataset from Hugging Face
            dataset = load_dataset("FronkonGames/steam-games-dataset")
            df = dataset["train"].to_pandas()
            
            logger.info(f"✅ Successfully loaded {len(df):,} Steam games")
            logger.info(f"📊 Dataset shape: {df.shape}")
            logger.info(f"🗂️  Columns: {list(df.columns)}")
            
            if save_raw:
                # Save raw dataset
                raw_file = self.raw_dir / "steam_games_raw.parquet"
                df.to_parquet(raw_file)
                logger.info(f"💾 Raw dataset saved to {raw_file}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading Steam games dataset: {str(e)}")
            raise
    
    def explore_dataset(self, df: pd.DataFrame) -> Dict:
        """Perform exploratory data analysis on the dataset"""
        logger.info("🔍 Starting dataset exploration...")
        
        analysis = {
            "basic_info": {
                "total_games": len(df),
                "columns": list(df.columns),
                "data_types": df.dtypes.to_dict(),
                "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
            },
            "missing_values": df.isnull().sum().to_dict(),
            "summary_stats": {}
        }
        
        # Analyze key columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if not df[col].isnull().all():
                analysis["summary_stats"][col] = {
                    "mean": float(df[col].mean()) if pd.notnull(df[col].mean()) else None,
                    "median": float(df[col].median()) if pd.notnull(df[col].median()) else None,
                    "min": float(df[col].min()) if pd.notnull(df[col].min()) else None,
                    "max": float(df[col].max()) if pd.notnull(df[col].max()) else None,
                    "unique_count": int(df[col].nunique())
                }
        
        # Analyze categorical columns
        categorical_analysis = {}
        for col in ['Genres', 'Categories', 'Tags', 'Developers', 'Publishers']:
            if col in df.columns:
                # Count non-null values
                non_null_count = df[col].notna().sum()
                categorical_analysis[col] = {
                    "non_null_count": int(non_null_count),
                    "null_percentage": float((len(df) - non_null_count) / len(df) * 100)
                }
                
                if non_null_count > 0:
                    # Sample some values
                    sample_values = df[col].dropna().head(5).tolist()
                    categorical_analysis[col]["sample_values"] = sample_values
        
        analysis["categorical_analysis"] = categorical_analysis
        
        # Save analysis
        analysis_file = self.processed_dir / "steam_dataset_analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        logger.info(f"📊 Dataset analysis saved to {analysis_file}")
        
        # Print key insights
        logger.info("🎯 Key Dataset Insights:")
        logger.info(f"   • Total games: {analysis['basic_info']['total_games']:,}")
        logger.info(f"   • Memory usage: {analysis['basic_info']['memory_usage']}")
        logger.info(f"   • Columns with data: {len(df.columns) - len([col for col, val in analysis['missing_values'].items() if val == len(df)])}")
        
        return analysis
    
    def preprocess_steam_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced preprocessing with advanced feature engineering"""
        logger.info("🧹 Starting ENHANCED data preprocessing...")
        
        # Create a copy for processing
        processed_df = df.copy()
        
        # 1. Enhanced text cleaning and standardization
        text_columns = ['Name', 'About the game', 'Developers', 'Publishers']
        for col in text_columns:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].astype(str).str.strip()
                # Replace various null representations
                processed_df[col] = processed_df[col].replace(['nan', 'None', '', 'null', 'N/A'], np.nan)
                # Clean up common text issues
                processed_df[col] = processed_df[col].str.replace(r'[^\x20-\x7E]', '', regex=True)  # Remove non-ASCII
        
        # 2. Advanced price processing and categorization
        if 'Price' in processed_df.columns:
            processed_df['Price'] = pd.to_numeric(processed_df['Price'], errors='coerce')
            processed_df['Is_Free'] = processed_df['Price'] == 0
            
            # Create price categories for better analysis
            processed_df['Price_Category'] = pd.cut(
                processed_df['Price'], 
                bins=[0, 0.01, 4.99, 9.99, 19.99, 39.99, np.inf],
                labels=['Free', 'Budget', 'Low', 'Mid', 'Premium', 'Luxury'],
                include_lowest=True
            )
            
            # Price percentile rankings
            processed_df['Price_Percentile'] = processed_df['Price'].rank(pct=True)
        
        # 3. Enhanced rating and popularity metrics
        rating_cols = ['Positive', 'Negative', 'Score rank', 'Recommendations']
        for col in rating_cols:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
        
        # Advanced rating calculations
        if 'Positive' in processed_df.columns and 'Negative' in processed_df.columns:
            processed_df['Total_Reviews'] = processed_df['Positive'].fillna(0) + processed_df['Negative'].fillna(0)
            processed_df['Review_Score'] = processed_df['Positive'] / (processed_df['Total_Reviews'] + 1)
            processed_df['Has_Reviews'] = processed_df['Total_Reviews'] > 0
            
            # Wilson confidence interval for better rating reliability
            processed_df['Wilson_Score'] = self._calculate_wilson_score(
                processed_df['Positive'].fillna(0), 
                processed_df['Total_Reviews'].fillna(0)
            )
            
            # Popularity metrics
            processed_df['Review_Density'] = np.log1p(processed_df['Total_Reviews'])  # Log-scaled review count
            processed_df['Controversy_Score'] = np.minimum(
                processed_df['Positive'].fillna(0), 
                processed_df['Negative'].fillna(0)
            ) / (processed_df['Total_Reviews'] + 1)
        
        # 4. Enhanced hierarchical categories with better parsing
        hierarchical_cols = ['Genres', 'Categories', 'Tags']
        for col in hierarchical_cols:
            if col in processed_df.columns:
                processed_df[f'{col}_List'] = processed_df[col].apply(self._enhanced_parse_list)
                processed_df[f'{col}_Count'] = processed_df[f'{col}_List'].apply(lambda x: len(x) if isinstance(x, list) else 0)
                
                # Create top-K binary encodings for most common items
                if f'{col}_List' in processed_df.columns:
                    self._create_top_k_binary_features(processed_df, f'{col}_List', col, k=10)
        
        # 5. Enhanced temporal features
        if 'Release date' in processed_df.columns:
            processed_df['Release_Date'] = pd.to_datetime(processed_df['Release date'], errors='coerce')
            processed_df['Release_Year'] = processed_df['Release_Date'].dt.year
            processed_df['Release_Month'] = processed_df['Release_Date'].dt.month
            processed_df['Release_Quarter'] = processed_df['Release_Date'].dt.quarter
            
            current_year = pd.Timestamp.now().year
            processed_df['Game_Age'] = current_year - processed_df['Release_Year']
            
            # Temporal categories
            processed_df['Era_Category'] = pd.cut(
                processed_df['Release_Year'],
                bins=[0, 2005, 2010, 2015, 2020, np.inf],
                labels=['Classic', 'Retro', 'Modern', 'Recent', 'Latest']
            )
        
        # 6. Enhanced platform and technical features
        platform_cols = ['Windows', 'Mac', 'Linux']
        if all(col in processed_df.columns for col in platform_cols):
            processed_df['Platform_Count'] = sum(processed_df[col].astype(bool).astype(int) for col in platform_cols)
            processed_df['Is_Multiplatform'] = processed_df['Platform_Count'] > 1
            processed_df['Platform_Exclusivity'] = processed_df['Platform_Count'] == 1
        
        # Technical feature indicators
        tech_features = ['Achievements', 'DLC count', 'Metacritic score']
        for col in tech_features:
            if col in processed_df.columns:
                processed_df[col] = pd.to_numeric(processed_df[col], errors='coerce')
                processed_df[f'Has_{col.replace(" ", "_")}'] = processed_df[col].notna() & (processed_df[col] > 0)
        
        # 7. Advanced text analysis for descriptions
        if 'About the game' in processed_df.columns:
            processed_df['Description_Length'] = processed_df['About the game'].str.len().fillna(0)
            processed_df['Description_Word_Count'] = processed_df['About the game'].str.split().str.len().fillna(0)
            processed_df['Has_Rich_Description'] = processed_df['Description_Word_Count'] >= 50
            
            # Sentiment indicators (basic)
            processed_df['Description_Sentiment'] = self._simple_sentiment_analysis(processed_df['About the game'])
        
        # 8. Enhanced developer/publisher analysis
        for col in ['Developers', 'Publishers']:
            if col in processed_df.columns:
                # Check if it's a major publisher/developer
                processed_df[f'Is_Major_{col[:-1]}'] = self._identify_major_entities(processed_df[col])
                processed_df[f'{col}_Count'] = processed_df[col].str.count(',') + 1  # Count multiple entities
        
        # 9. Comprehensive quality and completeness scoring
        processed_df['Data_Quality_Score'] = self._calculate_enhanced_quality_score(processed_df)
        processed_df['Content_Richness'] = self._calculate_content_richness(processed_df)
        processed_df['Commercial_Viability'] = self._calculate_commercial_viability(processed_df)
        
        # 10. Advanced filtering with multiple criteria
        high_quality_mask = (
            (processed_df['Data_Quality_Score'] >= 3) &  # Basic quality
            (processed_df['Name'].notna()) &  # Must have name
            (processed_df['Release_Year'] >= 1990)  # Reasonable release year
        )
        
        high_quality_games = processed_df[high_quality_mask].copy()
        
        logger.info(f"📊 Enhanced Preprocessing Results:")
        logger.info(f"   • Original games: {len(processed_df):,}")
        logger.info(f"   • High quality games: {len(high_quality_games):,}")
        logger.info(f"   • Filtered out: {len(processed_df) - len(high_quality_games):,}")
        logger.info(f"   • New features created: {len(high_quality_games.columns) - len(df.columns)}")
        
        # Save enhanced dataset
        processed_file = self.processed_dir / "steam_games_processed.parquet"
        high_quality_games.to_parquet(processed_file)
        logger.info(f"💾 Enhanced dataset saved to {processed_file}")
        
        return high_quality_games
    
    def _parse_list_string(self, value) -> List[str]:
        """Parse string representations of lists"""
        if pd.isna(value) or value == 'nan':
            return []
        
        try:
            # Handle different string formats
            if isinstance(value, str):
                # Remove extra whitespace and normalize
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    # Likely a string representation of a list
                    return ast.literal_eval(value)
                else:
                    # Split by common delimiters
                    for delimiter in [',', ';', '|']:
                        if delimiter in value:
                            return [item.strip() for item in value.split(delimiter) if item.strip()]
                    return [value]  # Single item
            return []
        except:
            return []
    
    def _enhanced_parse_list(self, value) -> List[str]:
        """Enhanced parsing with better cleaning and normalization"""
        if pd.isna(value) or value in ['nan', '', 'None', 'null']:
            return []
        
        try:
            if isinstance(value, list):
                return [str(item).strip() for item in value if str(item).strip()]
            
            if isinstance(value, str):
                value = value.strip()
                
                # Handle JSON-like strings
                if value.startswith('[') and value.endswith(']'):
                    try:
                        parsed = ast.literal_eval(value)
                        if isinstance(parsed, list):
                            return [str(item).strip() for item in parsed if str(item).strip()]
                    except:
                        # Remove brackets and try splitting
                        value = value[1:-1]
                
                # Split by various delimiters and clean
                for delimiter in [',', ';', '|', '/', '\\n', '\\t']:
                    if delimiter in value:
                        items = [item.strip().strip('"\'') for item in value.split(delimiter)]
                        return [item for item in items if item and item.lower() not in ['none', 'null', 'nan']]
                
                # Single item, clean it
                cleaned = value.strip().strip('"\'')
                if cleaned and cleaned.lower() not in ['none', 'null', 'nan']:
                    return [cleaned]
            
            return []
        except Exception:
            return []
    
    def _calculate_wilson_score(self, positive: pd.Series, total: pd.Series, z: float = 1.96) -> pd.Series:
        """Calculate Wilson confidence interval for better rating reliability"""
        # Avoid division by zero
        mask = total > 0
        result = pd.Series(0.0, index=positive.index)
        
        if mask.any():
            p = positive[mask] / total[mask]
            n = total[mask]
            
            # Wilson score formula
            center = p + (z * z) / (2 * n)
            spread = z * np.sqrt((p * (1 - p) + (z * z) / (4 * n)) / n)
            denominator = 1 + (z * z) / n
            
            result[mask] = (center - spread) / denominator
        
        return result
    
    def _create_top_k_binary_features(self, df: pd.DataFrame, list_col: str, base_name: str, k: int = 10):
        """Create binary features for top-K most common categories"""
        # Collect all items
        all_items = []
        for items in df[list_col].dropna():
            if isinstance(items, list):
                all_items.extend(items)
        
        if not all_items:
            return
        
        # Get top-K most common items
        from collections import Counter
        top_items = Counter(all_items).most_common(k)
        
        # Create binary columns
        for item, count in top_items:
            safe_name = re.sub(r'[^a-zA-Z0-9_]', '_', str(item))
            col_name = f'{base_name}_is_{safe_name}'
            
            df[col_name] = df[list_col].apply(
                lambda x: item in x if isinstance(x, list) else False
            )
    
    def _simple_sentiment_analysis(self, descriptions: pd.Series) -> pd.Series:
        """Basic sentiment analysis using keyword matching"""
        positive_words = {
            'amazing', 'excellent', 'fantastic', 'awesome', 'brilliant', 'outstanding',
            'incredible', 'wonderful', 'perfect', 'best', 'great', 'good', 'fun',
            'enjoyable', 'engaging', 'immersive', 'addictive', 'compelling'
        }
        
        negative_words = {
            'terrible', 'awful', 'horrible', 'worst', 'bad', 'boring', 'dull',
            'frustrating', 'disappointing', 'broken', 'buggy', 'glitchy', 'unplayable',
            'repetitive', 'tedious', 'confusing', 'poor', 'shallow'
        }
        
        def calculate_sentiment(text):
            if pd.isna(text):
                return 0.0
            
            text_lower = str(text).lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)
            
            total_sentiment_words = positive_count + negative_count
            if total_sentiment_words == 0:
                return 0.0  # Neutral
            
            return (positive_count - negative_count) / total_sentiment_words
        
        return descriptions.apply(calculate_sentiment)
    
    def _identify_major_entities(self, entities: pd.Series) -> pd.Series:
        """Identify major developers/publishers based on frequency"""
        # Count occurrences
        entity_counts = entities.value_counts()
        
        # Define threshold for "major" (top 5% or at least 20 games)
        threshold = max(20, int(len(entity_counts) * 0.05))
        major_entities = set(entity_counts[entity_counts >= threshold].index)
        
        return entities.apply(lambda x: str(x) in major_entities if pd.notna(x) else False)
    
    def _calculate_enhanced_quality_score(self, df: pd.DataFrame) -> pd.Series:
        """Calculate comprehensive data quality score (0-10)"""
        score = pd.Series(0, index=df.index)
        
        # Essential fields (4 points)
        essential_fields = ['Name', 'Release_Year', 'Genres', 'About the game']
        for field in essential_fields:
            if field in df.columns:
                score += df[field].notna().astype(int)
        
        # Important fields (3 points)
        important_fields = ['Price', 'Total_Reviews', 'Categories']
        for field in important_fields:
            if field in df.columns:
                score += df[field].notna().astype(int)
        
        # Nice-to-have fields (3 points)
        nice_fields = ['Developers', 'Publishers', 'Tags']
        for field in nice_fields:
            if field in df.columns:
                score += df[field].notna().astype(int)
        
        return score
    
    def _calculate_content_richness(self, df: pd.DataFrame) -> pd.Series:
        """Calculate content richness score (0-1)"""
        richness = pd.Series(0.0, index=df.index)
        
        # Description richness (0.3 weight)
        if 'Description_Word_Count' in df.columns:
            richness += np.minimum(df['Description_Word_Count'] / 100, 1.0) * 0.3
        
        # Category diversity (0.3 weight)
        category_cols = ['Genres_Count', 'Categories_Count', 'Tags_Count']
        for col in category_cols:
            if col in df.columns:
                richness += np.minimum(df[col] / 5, 1.0) * 0.1
        
        # Media richness (0.2 weight)
        if 'Screenshots' in df.columns:
            has_screenshots = df['Screenshots'].notna() & (df['Screenshots'] != '')
            richness += has_screenshots.astype(float) * 0.1
        
        if 'Movies' in df.columns:
            has_movies = df['Movies'].notna() & (df['Movies'] != '')
            richness += has_movies.astype(float) * 0.1
        
        # Review engagement (0.2 weight)
        if 'Review_Density' in df.columns:
            max_density = df['Review_Density'].quantile(0.95)
            if max_density > 0:
                richness += np.minimum(df['Review_Density'] / max_density, 1.0) * 0.2
        
        return richness
    
    def _calculate_commercial_viability(self, df: pd.DataFrame) -> pd.Series:
        """Calculate commercial viability score (0-1)"""
        viability = pd.Series(0.0, index=df.index)
        
        # Price reasonableness (0.2 weight)
        if 'Price' in df.columns:
            # Not free, not overpriced
            reasonable_price = (df['Price'] > 0) & (df['Price'] <= 60)
            viability += reasonable_price.astype(float) * 0.2
        
        # Review quality (0.3 weight)
        if 'Review_Score' in df.columns:
            viability += df['Review_Score'].fillna(0) * 0.3
        
        # Review quantity (0.2 weight)
        if 'Total_Reviews' in df.columns:
            # Normalize review count (log scale)
            log_reviews = np.log1p(df['Total_Reviews'])
            max_log_reviews = log_reviews.quantile(0.95)
            if max_log_reviews > 0:
                viability += np.minimum(log_reviews / max_log_reviews, 1.0) * 0.2
        
        # Platform availability (0.1 weight)
        if 'Platform_Count' in df.columns:
            viability += np.minimum(df['Platform_Count'] / 3, 1.0) * 0.1
        
        # Recency bonus (0.2 weight)
        if 'Game_Age' in df.columns:
            # Games from last 5 years get higher viability
            recency_score = np.maximum(0, (5 - df['Game_Age']) / 5)
            viability += recency_score * 0.2
        
        return viability
    
    def create_feature_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced feature matrix for machine learning"""
        logger.info("🔧 Creating ENHANCED feature matrix for ML...")
        
        # Enhanced feature selection including new features
        core_features = [
            'Price', 'Review_Score', 'Wilson_Score', 'Total_Reviews', 
            'Review_Density', 'Controversy_Score', 'Game_Age',
            'Platform_Count', 'Genres_Count', 'Categories_Count', 'Tags_Count'
        ]
        
        quality_features = [
            'Data_Quality_Score', 'Content_Richness', 'Commercial_Viability'
        ]
        
        text_features = [
            'Description_Length', 'Description_Word_Count', 'Description_Sentiment'
        ]
        
        boolean_features = [
            'Is_Free', 'Has_Reviews', 'Is_Multiplatform', 'Has_Rich_Description',
            'Has_Achievements', 'Has_DLC_count', 'Has_Metacritic_score',
            'Is_Major_Developer', 'Is_Major_Publisher'
        ]
        
        categorical_features = [
            'Price_Category', 'Era_Category', 'Price_Percentile'
        ]
        
        # Collect all available features
        all_feature_groups = [core_features, quality_features, text_features, boolean_features]
        all_features = []
        
        for feature_group in all_feature_groups:
            available = [col for col in feature_group if col in df.columns]
            all_features.extend(available)
            logger.info(f"   • {len(available)}/{len(feature_group)} features from {feature_group[0].split('_')[0]} group")
        
        # Add available categorical features
        available_categorical = [col for col in categorical_features if col in df.columns]
        all_features.extend(available_categorical)
        
        # Add binary features for top genres/categories
        binary_feature_cols = [col for col in df.columns if any(
            col.startswith(f'{base}_is_') for base in ['Genres', 'Categories', 'Tags']
        )]
        all_features.extend(binary_feature_cols[:30])  # Limit to top 30 binary features
        
        # Create feature matrix
        base_cols = ['AppID', 'Name']
        feature_matrix = df[base_cols + all_features].copy()
        
        logger.info(f"   • Total features selected: {len(all_features)}")
        
        # Enhanced missing value handling
        numeric_cols = feature_matrix.select_dtypes(include=[np.number]).columns
        boolean_cols = feature_matrix.select_dtypes(include=[bool]).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if col not in ['AppID']:  # Don't fill ID columns
                if feature_matrix[col].isnull().sum() > 0:
                    fill_value = feature_matrix[col].median()
                    feature_matrix[col] = feature_matrix[col].fillna(fill_value)
        
        # Fill boolean missing values with False
        for col in boolean_cols:
            feature_matrix[col] = feature_matrix[col].fillna(False)
        
        # Handle categorical features
        for col in available_categorical:
            if col in feature_matrix.columns:
                if feature_matrix[col].dtype.name == 'category':
                    # Fill missing categorical with most frequent
                    most_frequent = feature_matrix[col].mode()
                    if len(most_frequent) > 0:
                        feature_matrix[col] = feature_matrix[col].fillna(most_frequent[0])
        
        # Enhanced normalization
        try:
            from sklearn.preprocessing import StandardScaler, RobustScaler
            
            # Use RobustScaler for features that might have outliers
            outlier_prone_features = ['Total_Reviews', 'Price', 'Description_Length', 'Review_Density']
            robust_features = [col for col in outlier_prone_features if col in feature_matrix.columns]
            
            standard_features = [col for col in numeric_cols 
                               if col not in ['AppID'] + robust_features]
            
            if robust_features:
                robust_scaler = RobustScaler()
                feature_matrix[robust_features] = robust_scaler.fit_transform(feature_matrix[robust_features])
                logger.info(f"   • Applied robust scaling to {len(robust_features)} outlier-prone features")
            
            if standard_features:
                standard_scaler = StandardScaler()
                feature_matrix[standard_features] = standard_scaler.fit_transform(feature_matrix[standard_features])
                logger.info(f"   • Applied standard scaling to {len(standard_features)} features")
        
        except ImportError:
            logger.warning("   • Scikit-learn not available, skipping advanced scaling")
            # Fallback to simple normalization
            scaling_cols = [col for col in numeric_cols if col not in ['AppID']]
            for col in scaling_cols:
                std = feature_matrix[col].std()
                if std > 0:
                    feature_matrix[col] = (feature_matrix[col] - feature_matrix[col].mean()) / std
        
        # Save enhanced feature matrix
        feature_file = self.processed_dir / "steam_feature_matrix.parquet"
        feature_matrix.to_parquet(feature_file)
        logger.info(f"🎯 Enhanced feature matrix saved to {feature_file}")
        logger.info(f"   • Shape: {feature_matrix.shape}")
        logger.info(f"   • Features: {feature_matrix.shape[1] - 2}")  # Minus AppID and Name
        
        return feature_matrix

def main():
    """Main function to run the dataset loading and preprocessing"""
    logger.info("🚀 Starting GameMatch Dataset Processing Pipeline")
    
    # Initialize loader
    loader = GameMatchDataLoader()
    
    # Load Steam games dataset
    steam_df = loader.load_steam_games()
    
    # Perform exploration
    analysis = loader.explore_dataset(steam_df)
    
    # Preprocess the data
    processed_df = loader.preprocess_steam_data(steam_df)
    
    # Create feature matrix
    feature_matrix = loader.create_feature_matrix(processed_df)
    
    logger.info("✅ Dataset processing pipeline completed successfully!")
    logger.info(f"📈 Final dataset: {len(processed_df):,} games ready for ML")
    
    return processed_df, feature_matrix, analysis

if __name__ == "__main__":
    processed_data, features, analysis = main()