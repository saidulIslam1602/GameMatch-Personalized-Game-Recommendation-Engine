"""
GameMatch OpenAI Fine-tuning Module
Handles OpenAI fine-tuning for personalized game recommendations
"""

import pandas as pd
import numpy as np
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from openai import OpenAI
import time
from collections import defaultdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GameMatchFineTuner:
    """OpenAI fine-tuning pipeline for GameMatch recommendation system"""
    
    def __init__(self, data_dir="data", config_dir="config"):
        # Handle relative paths from different working directories
        project_root = Path(__file__).parent.parent.parent
        self.data_dir = project_root / data_dir if not Path(data_dir).is_absolute() else Path(data_dir)
        self.config_dir = project_root / config_dir if not Path(config_dir).is_absolute() else Path(config_dir)
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = Path("src/models")
        
        # Anti-overfitting measures
        self.game_usage_counter = {}
        self.max_game_usage = 5  # Maximum times a game can be used in training
        
        # Load OpenAI API key
        self.api_key = self._load_api_key()
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize templates
        self._init_templates()
    
    def reset_game_usage_counter(self):
        """Reset the game usage counter for fresh training data generation"""
        logger.info("🔄 Resetting game usage counter for balanced data generation")
        self.game_usage_counter = {}
    
    def get_usage_stats(self):
        """Get current game usage statistics"""
        if not self.game_usage_counter:
            return "No games used yet"
        
        top_used = sorted(self.game_usage_counter.items(), key=lambda x: x[1], reverse=True)[:10]
        stats = "Top 10 most used games:\n"
        for game, count in top_used:
            stats += f"   • {game}: {count} times\n"
        
        total_unique = len(self.game_usage_counter)
        total_uses = sum(self.game_usage_counter.values())
        stats += f"\nTotal unique games: {total_unique}, Total uses: {total_uses}"
        
        return stats
    
    def _init_templates(self):
        """Initialize all template sets for training data generation"""
        # Enhanced training data templates with context and personalization
        self.recommendation_templates = [
            "Recommend games similar to {game_name}",
            "Find games like {game_name} for me",
            "What games would you suggest if I liked {game_name}?",
            "I enjoyed {game_name}, what should I play next?",
            "Games similar to {game_name} please",
            "Suggest games for someone who loves {game_name}",
            "I'm a big fan of {game_name}. What other games would I enjoy?",
            "Based on my love for {game_name}, what are 3-5 must-play similar games?",
            "If {game_name} is my favorite game, what should be on my wishlist?",
            "I want games with the same vibe as {game_name}",
            "Looking for hidden gems similar to {game_name}",
        ]
        
        self.genre_templates = [
            "What are the best {genre} games?",
            "Recommend top {genre} games",
            "I want to play {genre} games, what do you suggest?",
            "Best {genre} games to play?",
            "Good {genre} game recommendations?",
            "I'm new to {genre} games, where should I start?",
            "What are some underrated {genre} games worth playing?",
            "Looking for the most innovative {genre} games",
            "I love {genre} - surprise me with something unique",
            "What {genre} games have the best storylines?",
        ]
        
        # Advanced contextual templates
        self.contextual_templates = [
            "I have {time_budget} hours to play this weekend. Recommend some {preferences} games.",
            "Looking for {preferences} games that work well on {platform}",
            "I enjoy {genre1} and {genre2}. What games combine both?",
            "Need {preferences} games for a {age_group} gamer",
            "What are some {preferences} co-op games for {player_count} players?",
            "Looking for {preferences} games with great {feature} mechanics",
            "I'm burned out on {overplayed_genre}. Suggest something different but similar quality",
            "What {preferences} games are trending right now?",
        ]
        
        # Multi-turn conversation starters
        self.conversation_starters = [
            "I'm looking for a new game to play",
            "Can you help me find my next favorite game?", 
            "I need game recommendations",
            "What should I play next?",
            "I'm bored with my current games",
        ]
    
    def _load_api_key(self) -> Optional[str]:
        """Load OpenAI API key from config file"""
        try:
            key_file = self.config_dir / "openai_key.txt"
            if key_file.exists():
                with open(key_file, 'r') as f:
                    return f.read().strip()
            else:
                logger.error("❌ OpenAI API key file not found!")
                return None
        except Exception as e:
            logger.error(f"❌ Error loading API key: {e}")
            return None
    
    def load_processed_data(self) -> pd.DataFrame:
        """Load the processed Steam games dataset"""
        logger.info("📂 Loading processed Steam games dataset...")
        
        processed_file = self.processed_dir / "steam_games_processed.parquet"
        
        if not processed_file.exists():
            raise FileNotFoundError("Processed dataset not found. Run dataset_loader.py first.")
        
        df = pd.read_parquet(processed_file)
        logger.info(f"✅ Loaded {len(df):,} processed games")
        
        return df
    
    def create_enhanced_similarity_examples(self, df: pd.DataFrame, num_examples=500) -> List[Dict]:
        """Create training examples for game similarity recommendations"""
        logger.info(f"🎯 Creating {num_examples} game similarity training examples...")
        
        examples = []
        
        # BALANCED SAMPLING: Use different quality tiers to prevent overfitting
        high_quality = df[
            (df['Data_Quality_Score'] >= 8) & 
            (df['Total_Reviews'] >= 100) &
            (df['Review_Score'] >= 0.8)
        ].copy()
        
        medium_quality = df[
            (df['Data_Quality_Score'] >= 6) & 
            (df['Total_Reviews'] >= 20) &
            (df['Review_Score'] >= 0.6) &
            (df['Review_Score'] < 0.8)
        ].copy()
        
        diverse_games = df[
            (df['Data_Quality_Score'] >= 4) & 
            (df['Total_Reviews'] >= 5) &
            (df['Review_Score'] >= 0.4)
        ].copy()
        
        # Sample from each tier to ensure diversity
        high_sample_size = min(len(high_quality), num_examples // 3)
        medium_sample_size = min(len(medium_quality), num_examples // 3) 
        diverse_sample_size = min(len(diverse_games), num_examples - high_sample_size - medium_sample_size)
        
        high_sample = high_quality.sample(high_sample_size, random_state=42) if high_sample_size > 0 else pd.DataFrame()
        medium_sample = medium_quality.sample(medium_sample_size, random_state=43) if medium_sample_size > 0 else pd.DataFrame()
        diverse_sample = diverse_games.sample(diverse_sample_size, random_state=44) if diverse_sample_size > 0 else pd.DataFrame()
        
        quality_games = pd.concat([high_sample, medium_sample, diverse_sample]).drop_duplicates(subset=['AppID']).copy()
        
        logger.info(f"🎮 Using {len(quality_games):,} balanced games for training")
        logger.info(f"   • High quality: {len(high_sample)}, Medium: {len(medium_sample)}, Diverse: {len(diverse_sample)}")
        
        for i in range(num_examples):
            try:
                # Select a random game as the base
                base_game = quality_games.sample(n=1).iloc[0]
                
                # Find similar games based on genres, categories, and price range
                similar_games = self._find_similar_games(base_game, quality_games, top_k=5)
                
                if len(similar_games) >= 3:
                    # Random template
                    template = random.choice(self.recommendation_templates)
                    user_prompt = template.format(game_name=base_game['Name'])
                    
                    # Create recommendation response
                    recommendations = self._create_recommendation_response(
                        base_game, similar_games[:3], quality_games
                    )
                    
                    example = {
                        "messages": [
                            {
                                "role": "system", 
                                "content": "You are GameMatch, a sophisticated game recommendation AI specialized in personalized gaming experiences. Analyze user preferences deeply, consider gameplay mechanics, narrative themes, visual styles, and community aspects. Provide detailed reasoning for each recommendation, explaining specific similarities and unique selling points. Always prioritize user satisfaction and discovery of perfect-fit games."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": recommendations
                            }
                        ]
                    }
                    
                    examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} similarity training examples")
        return examples
    
    def create_genre_recommendation_examples(self, df: pd.DataFrame, num_examples=200) -> List[Dict]:
        """Create training examples for genre-based recommendations"""
        logger.info(f"🎯 Creating {num_examples} genre recommendation training examples...")
        
        examples = []
        
        # Get top genres
        all_genres = []
        for genres_list in df['Genres_List'].dropna():
            if isinstance(genres_list, list):
                all_genres.extend(genres_list)
            elif isinstance(genres_list, str):
                # Handle string representations of lists
                try:
                    parsed_genres = eval(genres_list) if genres_list.startswith('[') else [genres_list]
                    if isinstance(parsed_genres, list):
                        all_genres.extend(parsed_genres)
                except:
                    all_genres.append(genres_list)
        
        if not all_genres:
            logger.warning("⚠️ No genres found in data, using fallback genres")
            all_genres = ['Action', 'Adventure', 'Strategy', 'RPG', 'Simulation', 'Sports', 'Racing', 'Puzzle', 'Platformer', 'Shooter']
        
        genre_counts = pd.Series(all_genres).value_counts()
        top_genres = genre_counts.head(20).index.tolist()
        
        logger.info(f"🏷️ Using top {len(top_genres)} genres for training: {top_genres[:5]}")
        
        for i in range(num_examples):
            try:
                # Select random genre
                genre = random.choice(top_genres)
                
                # Find best games in this genre
                genre_games = df[
                    df['Genres_List'].apply(
                        lambda x: isinstance(x, list) and genre in x if pd.notna(x) else False
                    ) &
                    (df['Data_Quality_Score'] >= 7) &
                    (df['Review_Score'] >= 0.7)
                ].sort_values(['Review_Score', 'Total_Reviews'], ascending=False)
                
                if len(genre_games) >= 3:
                    top_games = genre_games.head(5)
                    
                    # Random template
                    template = random.choice(self.genre_templates)
                    user_prompt = template.format(genre=genre)
                    
                    # Create genre recommendation response
                    response = self._create_genre_response(genre, top_games)
                    
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are GameMatch, an expert game recommendation AI. Provide personalized game recommendations with detailed reasoning based on genres, gameplay mechanics, themes, and user preferences. Always include specific details about why games are excellent in their genre."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": response
                            }
                        ]
                    }
                    
                    examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating genre example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} genre training examples")
        return examples
    
    def create_personalized_examples(self, df: pd.DataFrame, num_examples=300) -> List[Dict]:
        """Create training examples for personalized recommendations"""
        logger.info(f"🎯 Creating {num_examples} personalized recommendation examples...")
        
        examples = []
        
        # Create user preference scenarios
        preference_scenarios = [
            {"price_range": "budget", "max_price": 10, "preferences": ["high_rated", "indie"]},
            {"price_range": "free", "max_price": 0, "preferences": ["multiplayer", "popular"]},
            {"price_range": "premium", "max_price": 100, "preferences": ["aaa", "story_rich"]},
            {"price_range": "mid", "max_price": 30, "preferences": ["strategy", "simulation"]},
        ]
        
        for i in range(num_examples):
            try:
                scenario = random.choice(preference_scenarios)
                
                # Create user query
                user_queries = [
                    f"I'm looking for {scenario['price_range']} games under ${scenario['max_price']}",
                    f"Recommend some {scenario['price_range']} games for me",
                    f"What are good games under ${scenario['max_price']}?",
                    f"I want {scenario['price_range']} game recommendations",
                ]
                
                user_prompt = random.choice(user_queries)
                
                # Filter games based on scenario
                filtered_games = df[
                    (df['Price'] <= scenario['max_price']) &
                    (df['Data_Quality_Score'] >= 7) &
                    (df['Review_Score'] >= 0.6)
                ].sort_values(['Commercial_Viability', 'Review_Score'], ascending=False)
                
                if len(filtered_games) >= 3:
                    selected_games = filtered_games.head(4)
                    
                    response = self._create_personalized_response(
                        scenario, selected_games, user_prompt
                    )
                    
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are GameMatch, an expert game recommendation AI. Provide personalized game recommendations based on user preferences, budget constraints, and gaming interests. Always consider price, quality, and user requirements when making recommendations."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": response
                            }
                        ]
                    }
                    
                    examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating personalized example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} personalized training examples")
        return examples
    
    def create_enhanced_genre_examples(self, df: pd.DataFrame, num_examples=100) -> List[Dict]:
        """Create enhanced genre-based recommendation examples with better data handling"""
        logger.info(f"🎯 Creating {num_examples} ENHANCED genre recommendation examples...")
        
        examples = []
        
        # Define comprehensive genre mappings with fallbacks
        genre_mappings = {
            'Action': ['Action', 'FPS', 'Fighting', 'Hack and Slash', 'Combat'],
            'Adventure': ['Adventure', 'Exploration', 'Story Rich', 'Walking Simulator'],
            'Strategy': ['Strategy', 'RTS', 'Turn-Based Strategy', 'Grand Strategy', 'Tower Defense'],
            'RPG': ['RPG', 'JRPG', 'Action RPG', 'Character Development', 'Fantasy'],
            'Simulation': ['Simulation', 'Life Sim', 'Management', 'Building', 'City Builder'],
            'Sports': ['Sports', 'Racing', 'Football', 'Basketball', 'Golf'],
            'Puzzle': ['Puzzle', 'Logic', 'Physics', 'Match-3', 'Brain Training'],
            'Horror': ['Horror', 'Psychological Horror', 'Survival Horror', 'Thriller'],
            'Indie': ['Indie', 'Artistic', 'Experimental', 'Unique', 'Creative'],
            'Multiplayer': ['Multiplayer', 'Co-op', 'Online', 'Competitive', 'Team-Based']
        }
        
        for i in range(num_examples):
            try:
                # Select genre from mappings
                main_genre = random.choice(list(genre_mappings.keys()))
                genre_keywords = genre_mappings[main_genre]
                
                # Find games matching this genre using multiple criteria
                matching_games = df[df['Genres_List'].astype(str).str.contains('|'.join(genre_keywords), case=False, na=False)]
                
                if len(matching_games) == 0:
                    # Fallback: use any games with genre in tags or categories
                    matching_games = df[
                        (df['Tags_List'].astype(str).str.contains('|'.join(genre_keywords), case=False, na=False)) |
                        (df['Categories_List'].astype(str).str.contains('|'.join(genre_keywords), case=False, na=False))
                    ]
                
                if len(matching_games) >= 3:
                    # Select top games by quality and reviews
                    top_games = matching_games[
                        (matching_games['Data_Quality_Score'] >= 6) &
                        (matching_games['Review_Score'] >= 0.6)
                    ].sort_values(['Commercial_Viability', 'Wilson_Score'], ascending=False).head(5)
                    
                    if len(top_games) >= 3:
                        template = random.choice(self.genre_templates)
                        user_prompt = template.format(genre=main_genre)
                        
                        response = self._create_enhanced_genre_response(main_genre, top_games[:4])
                        
                        example = {
                            "messages": [
                                {
                                    "role": "system",
                                    "content": "You are GameMatch, an expert game curator with deep knowledge of gaming genres, mechanics, and player psychology. Provide genre-specific recommendations that highlight what makes each game exceptional in its category. Consider both mainstream hits and hidden gems."
                                },
                                {
                                    "role": "user",
                                    "content": user_prompt
                                },
                                {
                                    "role": "assistant",
                                    "content": response
                                }
                            ]
                        }
                        
                        examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating enhanced genre example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} enhanced genre training examples")
        return examples

    def create_enhanced_personalized_examples(self, df: pd.DataFrame, num_examples=200) -> List[Dict]:
        """Create enhanced personalized recommendations with diverse scenarios"""
        logger.info(f"🎯 Creating {num_examples} ENHANCED personalized examples...")
        
        examples = []
        
        # Expanded preference scenarios with real gaming contexts
        preference_scenarios = [
            {"price_range": "budget", "max_price": 10, "preferences": ["indie", "pixel_art"], "context": "student gamer"},
            {"price_range": "free", "max_price": 0, "preferences": ["multiplayer", "competitive"], "context": "casual player"},
            {"price_range": "premium", "max_price": 100, "preferences": ["AAA", "story_rich"], "context": "hardcore gamer"},
            {"price_range": "mid", "max_price": 30, "preferences": ["strategy", "complex"], "context": "strategy fan"},
            {"price_range": "sale", "max_price": 20, "preferences": ["discounted", "popular"], "context": "bargain hunter"},
            {"price_range": "new", "max_price": 60, "preferences": ["recent", "trending"], "context": "early adopter"},
            {"price_range": "retro", "max_price": 15, "preferences": ["classic", "nostalgia"], "context": "retro gamer"},
            {"price_range": "family", "max_price": 40, "preferences": ["family_friendly", "co-op"], "context": "family gaming"}
        ]
        
        for i in range(num_examples):
            try:
                scenario = random.choice(preference_scenarios)
                
                # Enhanced user queries with more context
                user_queries = [
                    f"I'm a {scenario['context']} looking for {scenario['price_range']} games under ${scenario['max_price']}",
                    f"As a {scenario['context']}, what {scenario['price_range']} games would you recommend?",
                    f"I have ${scenario['max_price']} budget and love {', '.join(scenario['preferences'])} games",
                    f"Perfect {scenario['price_range']} games for a {scenario['context']}?",
                    f"Recommend {scenario['price_range']} games under ${scenario['max_price']} that are {', '.join(scenario['preferences'])}"
                ]
                
                user_prompt = random.choice(user_queries)
                
                # Enhanced filtering with multiple criteria
                filtered_games = df[
                    (df['Price'] <= scenario['max_price']) &
                    (df['Data_Quality_Score'] >= 6) &
                    (df['Review_Score'] >= 0.5)
                ].copy()
                
                # Add preference-based scoring
                for pref in scenario['preferences']:
                    pref_mask = (
                        filtered_games['Genres_List'].astype(str).str.contains(pref, case=False, na=False) |
                        filtered_games['Tags_List'].astype(str).str.contains(pref, case=False, na=False) |
                        filtered_games['Categories_List'].astype(str).str.contains(pref, case=False, na=False)
                    )
                    filtered_games.loc[pref_mask, 'preference_score'] = filtered_games.loc[pref_mask].get('preference_score', 0) + 1
                
                # Sort by preference score and quality
                if 'preference_score' in filtered_games.columns:
                    filtered_games = filtered_games.sort_values(['preference_score', 'Commercial_Viability', 'Wilson_Score'], ascending=False)
                
                if len(filtered_games) >= 3:
                    selected_games = filtered_games.head(4)
                    
                    response = self._create_enhanced_personalized_response(scenario, selected_games, user_prompt)
                    
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are GameMatch, a personalized gaming advisor who understands diverse player types, budgets, and preferences. Tailor recommendations to specific user contexts, explain value propositions, and consider both explicit preferences and implicit gaming psychology."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": response
                            }
                        ]
                    }
                    
                    examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating enhanced personalized example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} enhanced personalized training examples")
        return examples

    def create_contextual_examples(self, df: pd.DataFrame, num_examples=150) -> List[Dict]:
        """Create contextual recommendation examples with specific scenarios"""
        logger.info(f"🎯 Creating {num_examples} contextual recommendation examples...")
        
        examples = []
        
        # Fixed context-template pairs to avoid formatting errors
        context_template_pairs = [
            # Time-based scenarios
            ({"time_budget": "2-3", "preferences": "quick session"}, 
             "I have 2-3 hours to play this weekend. Recommend some quick session games."),
            ({"time_budget": "10+", "preferences": "immersive"}, 
             "I have 10+ hours to play this weekend. Recommend some immersive games."),
            
            # Platform scenarios
            ({"preferences": "indie", "platform": "PC"}, 
             "Looking for indie games that work well on PC"),
            ({"preferences": "action", "platform": "console"}, 
             "Looking for action games that work well on console"),
            
            # Genre combination scenarios
            ({"genre1": "RPG", "genre2": "Strategy"}, 
             "I enjoy RPG and Strategy. What games combine both?"),
            ({"genre1": "Action", "genre2": "Adventure"}, 
             "I enjoy Action and Adventure. What games combine both?"),
            
            # Player count scenarios
            ({"preferences": "co-op", "player_count": "2"}, 
             "What are some co-op games for 2 players?"),
            ({"preferences": "multiplayer", "player_count": "4"}, 
             "What are some multiplayer games for 4 players?"),
            
            # Age group scenarios
            ({"preferences": "family-friendly", "age_group": "kids"}, 
             "Need family-friendly games for kids"),
            ({"preferences": "mature", "age_group": "adult"}, 
             "Need mature games for adult gamers"),
            
            # Feature-based scenarios
            ({"preferences": "building", "feature": "sandbox"}, 
             "Looking for building games with great sandbox mechanics"),
            ({"preferences": "strategy", "feature": "turn-based"}, 
             "Looking for strategy games with great turn-based mechanics"),
            
            # Burnout scenarios
            ({"preferences": "different", "overplayed_genre": "shooters"}, 
             "I'm burned out on shooters. Suggest something different but similar quality"),
            ({"preferences": "fresh", "overplayed_genre": "RPGs"}, 
             "I'm burned out on RPGs. Suggest something fresh but similar quality"),
        ]
        
        for i in range(num_examples):
            try:
                # Select a random context-template pair
                context, user_prompt = random.choice(context_template_pairs)
                
                # Filter games based on context
                context_games = self._filter_games_by_context(df, context)
                
                if len(context_games) >= 3:
                    selected_games = context_games.head(3)
                    response = self._create_contextual_response(context, selected_games)
                    
                    example = {
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are GameMatch, a context-aware gaming assistant. Consider user's time constraints, platform preferences, social context, and specific needs when recommending games. Explain why each recommendation fits their particular situation."
                            },
                            {
                                "role": "user",
                                "content": user_prompt
                            },
                            {
                                "role": "assistant",
                                "content": response
                            }
                        ]
                    }
                    
                    examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating contextual example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} contextual training examples")
        return examples

    def create_multiturn_examples(self, df: pd.DataFrame, num_examples=100) -> List[Dict]:
        """Create multi-turn conversation examples for natural dialogue"""
        logger.info(f"🎯 Creating {num_examples} multi-turn conversation examples...")
        
        examples = []
        
        for i in range(num_examples):
            try:
                # Start conversation
                starter = random.choice(self.conversation_starters)
                
                # Get initial recommendations
                sample_games = df[df['Data_Quality_Score'] >= 7].sample(n=3)
                
                initial_response = "I'd be happy to help you find your next great game! To give you the best recommendations, could you tell me:\n\n"
                initial_response += "• What genres do you typically enjoy?\n"
                initial_response += "• Any recent games you've loved?\n"
                initial_response += "• Your budget range?\n\n"
                initial_response += "In the meantime, here are some universally acclaimed games:\n\n"
                
                for idx, (_, game) in enumerate(sample_games.iterrows(), 1):
                    price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
                    initial_response += f"**{idx}. {game['Name']}** - {price_text}\n"
                    if pd.notna(game['Review_Score']):
                        initial_response += f"   • {game['Review_Score']*100:.0f}% positive reviews\n"
                    initial_response += "\n"
                
                # User follow-up
                followups = [
                    "I love RPGs and have a $30 budget",
                    "I enjoyed The Witcher 3, what's similar?",
                    "I want something co-op for me and my friend",
                    "Looking for something relaxing after work",
                    "I'm new to gaming, what should I try?"
                ]
                
                user_followup = random.choice(followups)
                
                # Create appropriate response based on follow-up
                followup_games = self._generate_followup_recommendations(df, user_followup)
                followup_response = self._create_followup_response(user_followup, followup_games)
                
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are GameMatch, a conversational game recommendation expert. Engage in natural dialogue, ask clarifying questions when needed, and build on previous conversation context to provide increasingly personalized recommendations."
                        },
                        {
                            "role": "user",
                            "content": starter
                        },
                        {
                            "role": "assistant",
                            "content": initial_response
                        },
                        {
                            "role": "user", 
                            "content": user_followup
                        },
                        {
                            "role": "assistant",
                            "content": followup_response
                        }
                    ]
                }
                
                examples.append(example)
            
            except Exception as e:
                logger.warning(f"⚠️ Error creating multiturn example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} multiturn conversation examples")
        return examples

    def create_structured_output_examples(self, df: pd.DataFrame, num_examples=100) -> List[Dict]:
        """Create examples with structured JSON outputs"""
        logger.info(f"🎯 Creating {num_examples} structured JSON output examples...")
        
        examples = []
        
        for i in range(num_examples):
            try:
                base_game = df[df['Data_Quality_Score'] >= 7].sample(n=1).iloc[0]
                similar_games = self._find_similar_games(base_game, df, top_k=3)
                
                user_prompt = f"Give me JSON recommendations for games similar to {base_game['Name']}"
                
                # Convert numpy arrays to lists for JSON serialization
                def safe_list_convert(item, max_items=3):
                    """Convert pandas/numpy objects to JSON-safe lists"""
                    if isinstance(item, (list, tuple)):
                        return list(item)[:max_items]
                    elif pd.isna(item) or item is None:
                        return []
                    elif isinstance(item, str):
                        try:
                            # Try to parse if it looks like a list string
                            if item.startswith('[') and item.endswith(']'):
                                parsed = eval(item)
                                return list(parsed)[:max_items] if isinstance(parsed, (list, tuple)) else [str(item)]
                        except:
                            pass
                        return [str(item)]
                    else:
                        return [str(item)]
                
                base_genres = safe_list_convert(base_game.get('Genres_List', []))
                
                # Create structured JSON response
                json_response = {
                    "query": f"Games similar to {base_game['Name']}",
                    "base_game_analysis": {
                        "name": str(base_game['Name']),
                        "genres": base_genres,
                        "price": float(base_game.get('Price', 0)),
                        "rating": f"{base_game.get('Review_Score', 0)*100:.0f}%" if pd.notna(base_game.get('Review_Score')) else "No rating"
                    },
                    "recommendations": [],
                    "reasoning": f"These games share core elements with {base_game['Name']} including similar genres, gameplay mechanics, and positive reception.",
                    "confidence_score": 0.85
                }
                
                for _, game in similar_games.iterrows():
                    game_genres = safe_list_convert(game.get('Genres_List', []))
                    
                    # Calculate genre overlap safely
                    genre_overlap = len(set(base_genres).intersection(set(game_genres)))
                    
                    recommendation = {
                        "name": str(game['Name']),
                        "price": float(game.get('Price', 0)),
                        "rating": f"{game.get('Review_Score', 0)*100:.0f}%" if pd.notna(game.get('Review_Score')) else "No rating",
                        "genres": game_genres,
                        "similarity_score": round(random.uniform(0.7, 0.95), 2),
                        "why_recommended": f"Shares {genre_overlap} genres with {base_game['Name']} and has similar gameplay appeal"
                    }
                    json_response["recommendations"].append(recommendation)
                
                json_string = json.dumps(json_response, indent=2)
                
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are GameMatch, an AI that provides structured game recommendations. When requested, output properly formatted JSON with detailed game information, similarity scores, and reasoning. Always include confidence scores and explain your recommendations."
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": json_string
                        }
                    ]
                }
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"⚠️ Error creating structured example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} structured output examples")
        return examples

    def create_evaluation_examples(self, df: pd.DataFrame, num_examples=50) -> List[Dict]:
        """Create examples that demonstrate evaluation and reasoning capabilities"""
        logger.info(f"🎯 Creating {num_examples} evaluation and reasoning examples...")
        
        examples = []
        evaluation_tasks = [
            "Compare and rank these games",
            "Analyze the pros and cons",
            "Which game offers better value",
            "Evaluate these recommendations",
            "Explain why this game is good"
        ]
        
        for i in range(num_examples):
            try:
                task = random.choice(evaluation_tasks)
                sample_games = df[df['Data_Quality_Score'] >= 7].sample(n=3)
                
                if "compare" in task.lower() or "rank" in task.lower():
                    game_names = [game['Name'] for _, game in sample_games.iterrows()]
                    user_prompt = f"{task}: {', '.join(game_names[:2])}"
                    response = self._create_comparison_response(sample_games.iloc[:2])
                    
                elif "analyze" in task.lower():
                    game = sample_games.iloc[0]
                    user_prompt = f"{task} of {game['Name']}"
                    response = self._create_analysis_response(game)
                    
                else:
                    game = sample_games.iloc[0]
                    user_prompt = f"{task}: {game['Name']}"
                    response = self._create_evaluation_response(game)
                
                example = {
                    "messages": [
                        {
                            "role": "system",
                            "content": "You are GameMatch, an analytical gaming expert capable of deep game evaluation, comparative analysis, and critical reasoning. Provide balanced, evidence-based assessments that consider multiple factors like gameplay, value, innovation, and player satisfaction."
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": response
                        }
                    ]
                }
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"⚠️ Error creating evaluation example {i}: {e}")
                continue
        
        logger.info(f"✅ Created {len(examples)} evaluation examples")
        return examples
    
    def _find_similar_games(self, base_game: pd.Series, games_df: pd.DataFrame, top_k=5) -> pd.DataFrame:
        """Find games similar to the base game with anti-overfitting measures"""
        import random
        
        # Calculate similarity based on multiple factors
        similarities = []
        
        base_genres = set(base_game['Genres_List']) if isinstance(base_game['Genres_List'], list) else set()
        base_categories = set(base_game['Categories_List']) if isinstance(base_game['Categories_List'], list) else set()
        base_price = base_game['Price']
        
        for idx, game in games_df.iterrows():
            if game['AppID'] == base_game['AppID']:  # Skip same game
                continue
            
            game_name = game['Name']
            
            # ANTI-OVERFITTING: Skip if game used too frequently
            if self.game_usage_counter.get(game_name, 0) >= self.max_game_usage:
                continue
            
            similarity_score = 0
            
            # Genre similarity
            game_genres = set(game['Genres_List']) if isinstance(game['Genres_List'], list) else set()
            if base_genres and game_genres:
                genre_similarity = len(base_genres.intersection(game_genres)) / len(base_genres.union(game_genres))
                similarity_score += genre_similarity * 0.4
            
            # Category similarity
            game_categories = set(game['Categories_List']) if isinstance(game['Categories_List'], list) else set()
            if base_categories and game_categories:
                category_similarity = len(base_categories.intersection(game_categories)) / len(base_categories.union(game_categories))
                similarity_score += category_similarity * 0.3
            
            # Price similarity (prefer similar price ranges)
            price_diff = abs(base_price - game['Price'])
            max_price = max(base_price, game['Price'])
            if max_price > 0:
                price_similarity = 1 - (price_diff / (max_price + 10))  # Add 10 to prevent division issues
                similarity_score += price_similarity * 0.2
            
            # BALANCED Quality bonus - reduced impact to prevent same games
            if game['Review_Score'] >= 0.7:
                similarity_score += 0.05  # Reduced from 0.1
            
            # DIVERSITY PENALTY: Reduce score for frequently used games
            usage_count = self.game_usage_counter.get(game_name, 0)
            diversity_penalty = usage_count * 0.02
            similarity_score -= diversity_penalty
            
            # Add randomness to break ties and increase diversity
            random_factor = random.uniform(-0.05, 0.05)
            similarity_score += random_factor
            
            similarities.append((idx, similarity_score))
        
        # If no games available due to usage limits, temporarily relax constraints
        if len(similarities) < top_k:
            temp_max = self.max_game_usage + 3
            for idx, game in games_df.iterrows():
                if game['AppID'] == base_game['AppID']:
                    continue
                
                game_name = game['Name']
                if self.game_usage_counter.get(game_name, 0) < temp_max:
                    # Add with baseline similarity to ensure we have enough games
                    similarity_score = random.uniform(0.3, 0.6)
                    similarities.append((idx, similarity_score))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in similarities[:min(top_k, len(similarities))]]
        
        result_games = games_df.loc[top_indices]
        
        # TRACK USAGE: Update counter for selected games
        for _, game in result_games.iterrows():
            game_name = game['Name']
            self.game_usage_counter[game_name] = self.game_usage_counter.get(game_name, 0) + 1
        
        return result_games
    
    def _create_recommendation_response(self, base_game: pd.Series, similar_games: pd.DataFrame, all_games: pd.DataFrame) -> str:
        """Create a detailed recommendation response"""
        
        base_genres = base_game['Genres_List'] if isinstance(base_game['Genres_List'], list) else []
        base_categories = base_game['Categories_List'] if isinstance(base_game['Categories_List'], list) else []
        
        response = f"Based on your interest in **{base_game['Name']}**, here are some excellent similar games:\n\n"
        
        for idx, (_, game) in enumerate(similar_games.iterrows(), 1):
            game_genres = game['Genres_List'] if isinstance(game['Genres_List'], list) else []
            shared_genres = list(set(base_genres).intersection(set(game_genres)))
            
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
            
            if shared_genres:
                response += f"   • Shared genres: {', '.join(shared_genres[:3])}\n"
            
            if pd.notna(game['Content_Richness']) and game['Content_Richness'] > 0.7:
                response += f"   • Rich content experience with detailed gameplay\n"
            
            response += "\n"
        
        # Add reasoning
        response += f"These recommendations are based on shared genres ({', '.join(base_genres[:3])}), "
        response += f"similar gameplay mechanics, and positive user reviews. "
        
        if base_game['Price'] > 0:
            response += f"They're also in a similar price range to {base_game['Name']}."
        
        return response.strip()
    
    def _create_genre_response(self, genre: str, top_games: pd.DataFrame) -> str:
        """Create a genre-based recommendation response"""
        
        response = f"Here are the top **{genre}** games I recommend:\n\n"
        
        for idx, (_, game) in enumerate(top_games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
            
            if pd.notna(game['Commercial_Viability']) and game['Commercial_Viability'] > 0.6:
                response += f"   • Highly recommended with strong commercial success\n"
            
            if pd.notna(game['Content_Richness']) and game['Content_Richness'] > 0.8:
                response += f"   • Rich, immersive {genre.lower()} experience\n"
            
            response += "\n"
        
        response += f"These {genre.lower()} games represent the best in the genre, "
        response += f"with excellent user ratings, engaging gameplay, and strong community support."
        
        return response.strip()
    
    def _create_personalized_response(self, scenario: Dict, games: pd.DataFrame, user_query: str) -> str:
        """Create a personalized recommendation response"""
        
        price_range = scenario['price_range']
        max_price = scenario['max_price']
        
        response = f"Here are excellent **{price_range}** games "
        if max_price > 0:
            response += f"under ${max_price} "
        response += "that I recommend:\n\n"
        
        for idx, (_, game) in enumerate(games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            genres = game['Genres_List'] if isinstance(game['Genres_List'], list) else []
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
            response += f"   • Genres: {', '.join(genres[:3])}\n"
            
            if pd.notna(game['Wilson_Score']) and game['Wilson_Score'] > 0.7:
                response += f"   • Highly reliable rating with consistent positive feedback\n"
            
            response += "\n"
        
        # Add value proposition
        if price_range == "free":
            response += "All these games offer excellent entertainment value at no cost, "
            response += "with active communities and regular updates."
        elif price_range == "budget":
            response += "These games offer exceptional value for money, "
            response += "providing hours of entertainment without breaking the bank."
        else:
            response += f"These {price_range} games represent excellent quality "
            response += "and are worth the investment for serious gamers."
        
        return response.strip()
    
    # Enhanced Helper Methods for Advanced Training Examples
    def _create_enhanced_genre_response(self, genre: str, top_games: pd.DataFrame) -> str:
        """Create an enhanced genre-based recommendation response"""
        
        response = f"Here are the best **{genre}** games I highly recommend:\n\n"
        
        for idx, (_, game) in enumerate(top_games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
            
            # Add genre-specific insights
            genres = game.get('Genres_List', [])
            if isinstance(genres, list) and len(genres) > 1:
                other_genres = [g for g in genres if g.lower() != genre.lower()][:2]
                if other_genres:
                    response += f"   • Also features: {', '.join(other_genres)}\n"
            
            # Add quality indicators
            if pd.notna(game.get('Commercial_Viability')) and game['Commercial_Viability'] > 0.7:
                response += f"   • Highly successful {genre.lower()} experience\n"
            
            if pd.notna(game.get('Content_Richness')) and game['Content_Richness'] > 0.8:
                response += f"   • Rich, polished gameplay with excellent {genre.lower()} mechanics\n"
            
            response += "\n"
        
        # Add genre-specific advice
        response += f"**Why these {genre} games stand out:**\n"
        response += f"Each game exemplifies what makes {genre.lower()} games compelling - whether through innovative mechanics, "
        response += f"engaging progression systems, or exceptional execution of {genre.lower()} fundamentals. "
        response += f"They're selected based on community reception, critical acclaim, and lasting appeal."
        
        return response.strip()
    
    def _create_enhanced_personalized_response(self, scenario: Dict, games: pd.DataFrame, user_prompt: str) -> str:
        """Create an enhanced personalized recommendation response"""
        
        response = f"Perfect {scenario['price_range']} recommendations for a {scenario['context']}!\n\n"
        
        for idx, (_, game) in enumerate(games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
            
            # Context-specific insights
            if scenario['context'] == 'student gamer':
                response += f"   • Great value for students - hours of entertainment per dollar\n"
            elif scenario['context'] == 'casual player':
                response += f"   • Easy to pick up and play, perfect for casual gaming\n"
            elif scenario['context'] == 'hardcore gamer':
                response += f"   • Deep, complex gameplay that rewards mastery\n"
            
            # Budget-specific messaging
            if game['Price'] == 0:
                response += f"   • Completely free - no upfront cost, just download and play\n"
            elif game['Price'] <= 10:
                response += f"   • Exceptional value - premium quality at budget price\n"
            
            # Preference matching
            preference_score = game.get('preference_score', 0)
            if preference_score > 0:
                response += f"   • Matches {preference_score} of your stated preferences\n"
            
            response += "\n"
        
        # Personalized conclusion
        response += f"**Why these work for you:**\n"
        response += f"As a {scenario['context']}, these games fit your {scenario['price_range']} budget while offering "
        response += f"{', '.join(scenario['preferences'])} elements you're looking for. Each provides excellent "
        response += f"entertainment value within your ${scenario['max_price']} price range."
        
        return response.strip()
    
    def _filter_games_by_context(self, df: pd.DataFrame, context: Dict) -> pd.DataFrame:
        """Filter games based on contextual requirements"""
        
        filtered = df[df['Data_Quality_Score'] >= 6].copy()
        
        # Time-based filtering
        if 'time_budget' in context:
            if '2-3' in context['time_budget']:
                # Quick session games - use Categories for arcade/casual games
                arcade_mask = filtered['Categories_List'].astype(str).str.contains('Arcade', case=False, na=False)
                casual_mask = filtered['Tags_List'].astype(str).str.contains('Casual', case=False, na=False)
                filtered = filtered[arcade_mask | casual_mask]
            elif '10+' in context['time_budget']:
                # Immersive games
                filtered = filtered[
                    (filtered['Content_Richness'] >= 0.7) |
                    filtered['Genres_List'].astype(str).str.contains('RPG|Strategy|Simulation', case=False, na=False)
                ]
        
        # Genre combination filtering
        if 'genre1' in context and 'genre2' in context:
            filtered = filtered[
                (filtered['Genres_List'].astype(str).str.contains(context['genre1'], case=False, na=False)) &
                (filtered['Genres_List'].astype(str).str.contains(context['genre2'], case=False, na=False))
            ]
        
        # Player count filtering
        if 'player_count' in context:
            if context['player_count'] == '2':
                coop_mask = filtered['Categories_List'].astype(str).str.contains('Co-op', case=False, na=False)
                local_mask = filtered['Tags_List'].astype(str).str.contains('Local', case=False, na=False)
                filtered = filtered[coop_mask | local_mask]
        
        # Preference-based filtering
        if 'preferences' in context:
            pref = context['preferences']
            if pref == 'relaxing':
                relaxing_mask = filtered['Tags_List'].astype(str).str.contains('Relaxing|Casual|Peaceful', case=False, na=False)
                filtered = filtered[relaxing_mask]
            elif pref == 'challenging':
                challenge_mask = filtered['Tags_List'].astype(str).str.contains('Difficult|Challenging|Hardcore', case=False, na=False)
                filtered = filtered[challenge_mask]
        
        # Return top results if we have any
        if len(filtered) > 0:
            return filtered.sort_values(['Commercial_Viability', 'Review_Score'], ascending=False)
        else:
            # Fallback to general high-quality games
            return df[df['Data_Quality_Score'] >= 6].sort_values('Wilson_Score', ascending=False)
    
    def _create_contextual_response(self, context: Dict, games: pd.DataFrame) -> str:
        """Create a contextual recommendation response"""
        
        response = "Based on your specific requirements, here are my tailored recommendations:\n\n"
        
        for idx, (_, game) in enumerate(games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • Rating: {rating_text}\n"
            
            # Context-specific explanations
            if 'time_budget' in context:
                if '2-3' in context['time_budget']:
                    response += f"   • Perfect for short gaming sessions - easy to pause anytime\n"
                elif '10+' in context['time_budget']:
                    response += f"   • Rich, immersive experience that rewards longer play sessions\n"
            
            if 'player_count' in context:
                response += f"   • Excellent for {context['player_count']} players - built for cooperation\n"
            
            if 'genre1' in context and 'genre2' in context:
                response += f"   • Unique blend of {context['genre1']} and {context['genre2']} mechanics\n"
            
            response += "\n"
        
        return response.strip()
    
    def _generate_followup_recommendations(self, df: pd.DataFrame, followup: str) -> pd.DataFrame:
        """Generate appropriate recommendations based on user followup"""
        
        # Default to high-quality games
        base_filter = df[df['Data_Quality_Score'] >= 7]
        
        if "RPG" in followup and "$30" in followup:
            return base_filter[(base_filter['Price'] <= 30) & 
                              (base_filter['Genres_List'].astype(str).str.contains('RPG', case=False, na=False))].sort_values('Wilson_Score', ascending=False).head(3)
        
        elif "Witcher 3" in followup or "Witcher" in followup:
            # Find RPG games with high story content
            rpg_games = base_filter[base_filter['Genres_List'].astype(str).str.contains('RPG', case=False, na=False)]
            return rpg_games[(rpg_games['Content_Richness'] >= 0.7)].sort_values('Commercial_Viability', ascending=False).head(3)
        
        elif "co-op" in followup:
            coop_games = base_filter[base_filter['Categories_List'].astype(str).str.contains('Co-op', case=False, na=False)]
            return coop_games.sort_values('Wilson_Score', ascending=False).head(3)
        
        elif "relaxing" in followup:
            relaxing_games = base_filter[base_filter['Tags_List'].astype(str).str.contains('Relaxing|Casual', case=False, na=False)]
            return relaxing_games.sort_values('Review_Score', ascending=False).head(3)
        
        else:
            # New to gaming - popular, easy games
            return base_filter[(base_filter['Commercial_Viability'] >= 0.7)].sort_values('Wilson_Score', ascending=False).head(3)
    
    def _create_followup_response(self, followup: str, games: pd.DataFrame) -> str:
        """Create appropriate follow-up response"""
        
        if "RPG" in followup:
            response = "Excellent choice! Here are some fantastic RPGs within your budget:\n\n"
        elif "Witcher" in followup:
            response = "Great taste! If you loved The Witcher 3, these games offer similar epic storytelling and immersive worlds:\n\n"
        elif "co-op" in followup:
            response = "Perfect for playing together! These co-op games are fantastic for friends:\n\n"
        elif "relaxing" in followup:
            response = "I understand the need to unwind! These games offer peaceful, stress-free experiences:\n\n"
        else:
            response = "Welcome to gaming! These are perfect starting points - accessible, polished, and universally loved:\n\n"
        
        for idx, (_, game) in enumerate(games.iterrows(), 1):
            price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
            rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
            
            response += f"**{idx}. {game['Name']}** - {price_text}\n"
            response += f"   • {rating_text} positive reviews\n"
            
            if "RPG" in followup:
                response += f"   • Deep character progression and engaging storyline\n"
            elif "Witcher" in followup:
                response += f"   • Rich narrative with meaningful choices and consequences\n"
            elif "co-op" in followup:
                response += f"   • Seamless cooperative gameplay designed for teamwork\n"
            
            response += "\n"
        
        return response.strip()
    
    def _create_comparison_response(self, games: pd.DataFrame) -> str:
        """Create a comparative analysis of games"""
        
        game1, game2 = games.iloc[0], games.iloc[1]
        
        response = f"**Comparing {game1['Name']} vs {game2['Name']}:**\n\n"
        
        # Price comparison
        response += "**Value Analysis:**\n"
        response += f"• {game1['Name']}: ${game1['Price']:.2f}\n"
        response += f"• {game2['Name']}: ${game2['Price']:.2f}\n"
        
        if game1['Price'] < game2['Price']:
            response += f"🏆 {game1['Name']} offers better upfront value\n\n"
        elif game2['Price'] < game1['Price']:
            response += f"🏆 {game2['Name']} offers better upfront value\n\n"
        else:
            response += "Equal pricing - value depends on personal preference\n\n"
        
        # Review comparison
        response += "**Community Reception:**\n"
        if pd.notna(game1.get('Review_Score')) and pd.notna(game2.get('Review_Score')):
            response += f"• {game1['Name']}: {game1['Review_Score']*100:.0f}% positive ({int(game1['Total_Reviews']):,} reviews)\n"
            response += f"• {game2['Name']}: {game2['Review_Score']*100:.0f}% positive ({int(game2['Total_Reviews']):,} reviews)\n"
            
            if game1['Review_Score'] > game2['Review_Score']:
                response += f"🏆 {game1['Name']} has better user satisfaction\n\n"
            else:
                response += f"🏆 {game2['Name']} has better user satisfaction\n\n"
        
        # Final recommendation
        response += "**Final Recommendation:**\n"
        if game1.get('Commercial_Viability', 0) > game2.get('Commercial_Viability', 0):
            response += f"For most players, I'd recommend **{game1['Name']}** due to its stronger overall appeal and proven track record."
        else:
            response += f"For most players, I'd recommend **{game2['Name']}** due to its stronger overall appeal and proven track record."
        
        return response.strip()
    
    def _create_analysis_response(self, game: pd.Series) -> str:
        """Create detailed analysis of a single game"""
        
        response = f"**Detailed Analysis: {game['Name']}**\n\n"
        
        # Basic info
        price_text = "Free" if game['Price'] == 0 else f"${game['Price']:.2f}"
        rating_text = f"{game['Review_Score']*100:.0f}%" if pd.notna(game['Review_Score']) else "No rating"
        
        response += f"**Overview:**\n"
        response += f"• Price: {price_text}\n"
        response += f"• User Rating: {rating_text} ({int(game['Total_Reviews']):,} reviews)\n"
        response += f"• Genres: {', '.join(game.get('Genres_List', [])[:4])}\n\n"
        
        # Strengths
        response += "**Strengths:**\n"
        if pd.notna(game.get('Review_Score')) and game['Review_Score'] >= 0.8:
            response += "✅ Excellent user satisfaction and critical reception\n"
        if pd.notna(game.get('Content_Richness')) and game['Content_Richness'] >= 0.7:
            response += "✅ Rich, polished content with high production values\n"
        if game['Price'] <= 20:
            response += "✅ Great value proposition for the price point\n"
        if pd.notna(game.get('Commercial_Viability')) and game['Commercial_Viability'] >= 0.7:
            response += "✅ Proven market success and broad appeal\n"
        
        # Bottom line
        response += f"\n**Bottom Line:**\n"
        if game.get('Data_Quality_Score', 0) >= 8:
            response += f"{game['Name']} is a standout title that delivers on its promises."
        else:
            response += f"{game['Name']} is a solid choice with some considerations."
        
        return response.strip()
    
    def _create_evaluation_response(self, game: pd.Series) -> str:
        """Create evaluation response explaining why a game is good"""
        
        response = f"**Why {game['Name']} is worth playing:**\n\n"
        
        # User reception
        if pd.notna(game.get('Review_Score')):
            if game['Review_Score'] >= 0.8:
                response += f"🌟 **Exceptional Reception**: {game['Review_Score']*100:.0f}% positive rating from {int(game['Total_Reviews']):,} players proves its quality\n\n"
            elif game['Review_Score'] >= 0.7:
                response += f"👍 **Strong Reception**: {game['Review_Score']*100:.0f}% positive rating shows solid player satisfaction\n\n"
        
        # Value proposition
        if game['Price'] == 0:
            response += "💰 **Incredible Value**: Completely free with no upfront cost\n\n"
        elif game['Price'] <= 15:
            response += f"💰 **Excellent Value**: Premium gaming experience at just ${game['Price']:.2f}\n\n"
        
        # Summary
        response += f"**Summary**: {game['Name']} combines solid gameplay fundamentals with "
        response += f"{'exceptional value' if game['Price'] <= 15 else 'premium quality'}, making it a "
        response += f"{'must-play' if game.get('Review_Score', 0) >= 0.8 else 'recommended'} title."
        
        return response.strip()
    
    def save_training_data(self, examples: List[Dict], filename: str):
        """Save training examples to JSONL format"""
        
        output_file = self.models_dir / filename
        
        logger.info(f"💾 Saving {len(examples)} training examples to {output_file}")
        
        with open(output_file, 'w') as f:
            for example in examples:
                f.write(json.dumps(example) + '\n')
        
        logger.info(f"✅ Training data saved successfully")
        
        # Validate the JSONL file
        self._validate_jsonl_file(output_file)
        
        return output_file
    
    def _validate_jsonl_file(self, file_path: Path):
        """Validate JSONL training file format"""
        logger.info(f"🔍 Validating JSONL file format...")
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            valid_count = 0
            for i, line in enumerate(lines, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Check required structure
                    if 'messages' in data and isinstance(data['messages'], list):
                        messages = data['messages']
                        if len(messages) >= 2:  # At least system/user or user/assistant
                            valid_count += 1
                        else:
                            logger.warning(f"⚠️ Line {i}: Insufficient messages")
                    else:
                        logger.warning(f"⚠️ Line {i}: Missing 'messages' field")
                
                except json.JSONDecodeError:
                    logger.error(f"❌ Line {i}: Invalid JSON")
            
            logger.info(f"✅ Validation complete: {valid_count}/{len(lines)} examples are valid")
            
        except Exception as e:
            logger.error(f"❌ Validation error: {e}")
    
    def create_complete_training_dataset(self) -> str:
        """Create the complete training dataset for GameMatch fine-tuning"""
        logger.info("🚀 Creating ENHANCED GameMatch training dataset...")
        
        # Load processed data
        df = self.load_processed_data()
        
        # Create different types of training examples with enhanced methods
        similarity_examples = self.create_enhanced_similarity_examples(df, num_examples=300)
        genre_examples = self.create_enhanced_genre_examples(df, num_examples=100)
        personalized_examples = self.create_enhanced_personalized_examples(df, num_examples=200)
        contextual_examples = self.create_contextual_examples(df, num_examples=150)
        multiturn_examples = self.create_multiturn_examples(df, num_examples=100)
        structured_examples = self.create_structured_output_examples(df, num_examples=100)
        evaluation_examples = self.create_evaluation_examples(df, num_examples=50)
        
        # Combine all examples
        all_examples = (similarity_examples + genre_examples + personalized_examples + 
                       contextual_examples + multiturn_examples + structured_examples + evaluation_examples)
        
        # Advanced shuffling with stratification to ensure balanced training
        random.shuffle(all_examples)
        
        logger.info(f"📊 ENHANCED Training Dataset Created: {len(all_examples)} examples")
        logger.info(f"   • Enhanced similarity recommendations: {len(similarity_examples)}")
        logger.info(f"   • Enhanced genre recommendations: {len(genre_examples)}")
        logger.info(f"   • Enhanced personalized recommendations: {len(personalized_examples)}")
        logger.info(f"   • Contextual recommendations: {len(contextual_examples)}")
        logger.info(f"   • Multi-turn conversations: {len(multiturn_examples)}")
        logger.info(f"   • Structured JSON outputs: {len(structured_examples)}")
        logger.info(f"   • Evaluation & reasoning: {len(evaluation_examples)}")
        
        # Save enhanced training data
        training_file = self.save_training_data(all_examples, "gamematch_enhanced_training_data.jsonl")
        
        return str(training_file)
    
    def upload_training_file(self, file_path: str) -> str:
        """Upload training file to OpenAI"""
        logger.info(f"📤 Uploading training file to OpenAI...")
        
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        try:
            with open(file_path, "rb") as f:
                response = self.client.files.create(
                    file=f,
                    purpose="fine-tune"
                )
            
            file_id = response.id
            logger.info(f"✅ Training file uploaded successfully: {file_id}")
            
            return file_id
        
        except Exception as e:
            logger.error(f"❌ Error uploading file: {e}")
            raise
    
    def start_fine_tuning(self, file_id: str, model="gpt-3.5-turbo-1106") -> str:
        """Start OpenAI fine-tuning job"""
        logger.info(f"🎯 Starting fine-tuning job with model: {model}")
        
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        try:
            response = self.client.fine_tuning.jobs.create(
                training_file=file_id,
                model=model,
                hyperparameters={
                    "n_epochs": 3,  # Good balance for our dataset size
                }
            )
            
            job_id = response.id
            logger.info(f"✅ Fine-tuning job started: {job_id}")
            logger.info(f"📊 Status: {response.status}")
            
            return job_id
        
        except Exception as e:
            logger.error(f"❌ Error starting fine-tuning: {e}")
            raise
    
    def check_job_status(self, job_id: str):
        """Check fine-tuning job status"""
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        try:
            response = self.client.fine_tuning.jobs.retrieve(job_id)
            
            logger.info(f"📊 Job Status: {response.status}")
            
            if response.status == "succeeded":
                logger.info(f"🎉 Fine-tuning completed! Model: {response.fine_tuned_model}")
                return response.fine_tuned_model
            elif response.status == "failed":
                logger.error(f"❌ Fine-tuning failed: {response.error}")
            elif response.status in ["validating_files", "queued", "running"]:
                logger.info(f"⏳ Fine-tuning in progress...")
            
            return response.status
        
        except Exception as e:
            logger.error(f"❌ Error checking job status: {e}")
            raise
    
    def test_fine_tuned_model(self, model_id: str, test_queries: List[str] = None):
        """Test the fine-tuned model with sample queries"""
        logger.info(f"🧪 Testing fine-tuned model: {model_id}")
        
        if not self.client:
            raise ValueError("OpenAI client not initialized. Check API key.")
        
        if not test_queries:
            test_queries = [
                "Recommend games similar to The Witcher 3",
                "What are the best strategy games?",
                "I'm looking for free multiplayer games",
                "Suggest some budget games under $10",
                "Games similar to Stardew Valley please"
            ]
        
        logger.info("🎮 Testing with sample queries:")
        
        for query in test_queries:
            try:
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "You are GameMatch, an expert game recommendation AI."},
                        {"role": "user", "content": query}
                    ],
                    max_tokens=300,
                    temperature=0.7
                )
                
                logger.info(f"\n📝 Query: {query}")
                logger.info(f"🤖 Response: {response.choices[0].message.content}")
                logger.info("-" * 80)
                
            except Exception as e:
                logger.error(f"❌ Error testing query '{query}': {e}")
    
    def run_complete_pipeline(self):
        """Run the complete fine-tuning pipeline"""
        logger.info("🚀 STARTING GAMEMATCH OPENAI FINE-TUNING PIPELINE")
        logger.info("=" * 60)
        
        try:
            # Step 1: Create training data
            training_file = self.create_complete_training_dataset()
            
            # Step 2: Upload to OpenAI
            file_id = self.upload_training_file(training_file)
            
            # Step 3: Start fine-tuning
            job_id = self.start_fine_tuning(file_id)
            
            logger.info(f"📋 FINE-TUNING SUMMARY:")
            logger.info(f"   • Training file: {training_file}")
            logger.info(f"   • OpenAI file ID: {file_id}")
            logger.info(f"   • Job ID: {job_id}")
            logger.info(f"   • Status: Check with check_job_status('{job_id}')")
            
            logger.info("✅ Fine-tuning pipeline initiated successfully!")
            logger.info("⏳ Fine-tuning typically takes 10-30 minutes.")
            logger.info(f"🔍 Monitor progress with: check_job_status('{job_id}')")
            
            return {
                "training_file": training_file,
                "file_id": file_id,
                "job_id": job_id,
                "status": "started"
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise

def main():
    """Main function to run the fine-tuning pipeline"""
    finetuner = GameMatchFineTuner()
    result = finetuner.run_complete_pipeline()
    return result

if __name__ == "__main__":
    result = main()