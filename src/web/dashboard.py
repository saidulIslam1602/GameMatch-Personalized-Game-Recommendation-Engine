#!/usr/bin/env python3
"""
GameMatch Visual Dashboard
Industry-standard web interface for game recommendations
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import hashlib
import secrets
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for, flash
from flask_cors import CORS
import logging
from datetime import datetime, timedelta
from pathlib import Path
import sqlite3

# Add project root to path for proper imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
        logger.info("Loaded environment variables from .env file")
except ImportError:
    logger.info("python-dotenv not available, using system environment variables only")

from models.openai_finetuning import GameMatchFineTuner
from data.dataset_loader import GameMatchDataLoader
from web.mssql_auth import MSSQLUserAuth

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')
CORS(app)

# Configure Flask with secure settings
# Use a consistent secret key to maintain sessions across restarts
secret_key = os.getenv('FLASK_SECRET_KEY')
if not secret_key:
    # Generate and save a persistent secret key
    secret_file = project_root / '.flask_secret'
    if secret_file.exists():
        secret_key = secret_file.read_text().strip()
    else:
        secret_key = secrets.token_urlsafe(32)
        secret_file.write_text(secret_key)
        logger.info("Generated new Flask secret key")

app.config['SECRET_KEY'] = secret_key
app.config['JSON_SORT_KEYS'] = False
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['SESSION_COOKIE_SECURE'] = os.getenv('ENVIRONMENT', 'development') == 'production'
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_NAME'] = 'gamematch_session'

class UserAuth:
    """User authentication and management system"""
    
    def __init__(self, db_path="data/users.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize user database"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                username TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Create user preferences table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                preferred_genres TEXT,
                preferred_price_range TEXT,
                preferred_rating_range TEXT,
                liked_games TEXT,
                disliked_games TEXT,
                search_history TEXT,
                interaction_count INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create game interactions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                interaction_type TEXT NOT NULL,
                interaction_data TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create game views table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS game_views (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                game_id INTEGER NOT NULL,
                view_duration INTEGER DEFAULT 0,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create user recommendations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                query TEXT NOT NULL,
                recommended_games TEXT NOT NULL,
                recommendation_type TEXT NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("✅ User database initialized")
    
    def hash_password(self, password):
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def verify_password(self, password, password_hash):
        """Verify password against hash"""
        try:
            salt, hash_part = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt.encode('utf-8'), 100000)
            return password_hash_check.hex() == hash_part
        except:
            return False
    
    def register_user(self, email, password, username):
        """Register a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if user already exists
            cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
            if cursor.fetchone():
                return False, "Email already registered"
            
            # Hash password and create user
            password_hash = self.hash_password(password)
            cursor.execute(
                "INSERT INTO users (email, password_hash, username) VALUES (?, ?, ?)",
                (email, password_hash, username)
            )
            
            user_id = cursor.lastrowid
            
            # Create default preferences
            cursor.execute(
                "INSERT INTO user_preferences (user_id) VALUES (?)",
                (user_id,)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ User registered: {email}")
            return True, "Registration successful"
            
        except Exception as e:
            logger.error(f"❌ Registration error: {e}")
            return False, "Registration failed"
    
    def authenticate_user(self, email, password):
        """Authenticate user login"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, password_hash, username FROM users WHERE email = ? AND is_active = 1",
                (email,)
            )
            user = cursor.fetchone()
            
            if user and self.verify_password(password, user[1]):
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?",
                    (user[0],)
                )
                conn.commit()
                conn.close()
                
                logger.info(f"✅ User authenticated: {email}")
                return True, {"user_id": user[0], "username": user[2], "email": email}
            else:
                conn.close()
                return False, "Invalid credentials"
                
        except Exception as e:
            logger.error(f"❌ Authentication error: {e}")
            return False, "Authentication failed"
    
    def get_user_preferences(self, user_id):
        """Get user preferences from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT * FROM user_preferences WHERE user_id = ?",
                (user_id,)
            )
            prefs = cursor.fetchone()
            conn.close()
            
            if prefs:
                return {
                    'user_id': user_id,
                    'preferred_genres': json.loads(prefs[2]) if prefs[2] else {},
                    'preferred_price_range': prefs[3],
                    'preferred_rating_range': prefs[4],
                    'liked_games': json.loads(prefs[5]) if prefs[5] else [],
                    'disliked_games': json.loads(prefs[6]) if prefs[6] else [],
                    'search_history': json.loads(prefs[7]) if prefs[7] else [],
                    'interaction_count': prefs[8] or 0,
                    'last_updated': prefs[9]
                }
            return None
            
        except Exception as e:
            logger.error(f"❌ Error getting user preferences: {e}")
            return None
    
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_preferences SET
                    preferred_genres = ?,
                    preferred_price_range = ?,
                    preferred_rating_range = ?,
                    liked_games = ?,
                    disliked_games = ?,
                    search_history = ?,
                    interaction_count = ?,
                    last_updated = CURRENT_TIMESTAMP
                WHERE user_id = ?
            ''', (
                json.dumps(preferences.get('preferred_genres', {})),
                preferences.get('preferred_price_range'),
                preferences.get('preferred_rating_range'),
                json.dumps(preferences.get('liked_games', [])),
                json.dumps(preferences.get('disliked_games', [])),
                json.dumps(preferences.get('search_history', [])),
                preferences.get('interaction_count', 0),
                user_id
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating user preferences: {e}")
            return False
    
    def track_game_interaction(self, user_id, game_id, interaction_type, interaction_data=None):
        """Track user interaction with a game"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO game_interactions (user_id, game_id, interaction_type, interaction_data)
                VALUES (?, ?, ?, ?)
            ''', (user_id, game_id, interaction_type, json.dumps(interaction_data) if interaction_data else None))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error tracking game interaction: {e}")
            return False
    
    def track_game_view(self, user_id, game_id, view_duration=0):
        """Track user viewing a game"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO game_views (user_id, game_id, view_duration)
                VALUES (?, ?, ?)
            ''', (user_id, game_id, view_duration))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error tracking game view: {e}")
            return False
    
    def save_recommendation(self, user_id, query, recommended_games, recommendation_type):
        """Save recommendation for analysis"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_recommendations (user_id, query, recommended_games, recommendation_type)
                VALUES (?, ?, ?, ?)
            ''', (user_id, query, json.dumps(recommended_games), recommendation_type))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving recommendation: {e}")
            return False
    
    def get_user_interactions(self, user_id, limit=100):
        """Get user's game interactions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT game_id, interaction_type, interaction_data, timestamp
                FROM game_interactions
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (user_id, limit))
            
            interactions = cursor.fetchall()
            conn.close()
            
            return [{
                'game_id': row[0],
                'interaction_type': row[1],
                'interaction_data': json.loads(row[2]) if row[2] else None,
                'timestamp': row[3]
            } for row in interactions]
            
        except Exception as e:
            logger.error(f"❌ Error getting user interactions: {e}")
            return []

class GameMatchDashboard:
    """Visual dashboard for GameMatch recommendations"""
    
    def __init__(self):
        self.finetuner = GameMatchFineTuner()
        self.dataset_loader = GameMatchDataLoader()
        # Use MSSQL authentication with correct password
        # Use environment-based authentication
        self.auth = MSSQLUserAuth()
        self.df = None
        self.user_profiles = {}  # Store user preferences and behavior
        self.session_data = {}   # Track current session interactions
        self.load_data()
    
    def clean_image_url(self, url):
        """Clean and validate image URL"""
        if not url or pd.isna(url) or str(url).lower() in ['nan', 'null', 'none', '']:
            return ''
        
        url = str(url).strip()
        
        # Remove quotes and clean URL
        url = url.replace('"', '').replace("'", '')
        
        # Check if it's a valid HTTP/HTTPS URL
        if url.startswith(('http://', 'https://')):
            return url
        
        # If it's a Steam CDN URL without protocol, add https
        if url.startswith('steamcdn-a.akamaihd.net') or url.startswith('cdn.akamai.steamstatic.com'):
            return 'https://' + url
        
        return ''
    
    def load_data(self):
        """Load processed game data from actual dataset"""
        try:
            # Load from the actual processed dataset using project root
            data_path = project_root / "data" / "processed" / "steam_games_processed.parquet"
            
            if not data_path.exists():
                logger.error(f"❌ Processed dataset not found at {data_path}")
                logger.info("💡 Please run data preprocessing first: python3 src/data/dataset_loader.py")
                self.df = pd.DataFrame()
                return
            
            self.df = pd.read_parquet(data_path)
            logger.info(f"✅ Loaded {len(self.df)} games from actual dataset")
            
            # Validate data quality
            if len(self.df) == 0:
                logger.warning("⚠️ Dataset is empty")
            else:
                logger.info(f"📊 Dataset info: {len(self.df)} games, {len(self.df.columns)} features")
                
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            self.df = pd.DataFrame()
    
    def get_game_recommendations(self, query, user_id=None, num_recommendations=5):
        """Get intelligent game recommendations based on user profile and query"""
        try:
            # Get or create user profile
            user_profile = self._get_or_create_user_profile(user_id)
            
            # Update user profile with current query
            self._update_user_profile(user_profile, query)
            
            # Choose recommendation strategy based on user profile
            if user_profile['interaction_count'] < 3:
                # Cold start: New user with minimal data
                recommendations = self._get_cold_start_recommendations(query, user_profile, num_recommendations)
            elif user_profile['interaction_count'] < 10:
                # Learning phase: Some data, hybrid approach
                recommendations = self._get_hybrid_recommendations(query, user_profile, num_recommendations)
            else:
                # Mature profile: Full personalization
                recommendations = self._get_personalized_recommendations(query, user_profile, num_recommendations)
            
            # Update user profile with interaction
            self._record_interaction(user_profile, query, recommendations)
            
            return recommendations
        except Exception as e:
            logger.error(f"❌ Error getting recommendations: {e}")
            return []
    
    def _get_or_create_user_profile(self, user_id):
        """Get or create user profile for personalization"""
        if user_id is None:
            user_id = "anonymous"
        
        # Try to get from database first
        if user_id != "anonymous":
            db_profile = self.auth.get_user_preferences(user_id)
            if db_profile:
                # Convert database profile to our format
                profile = {
                    'user_id': user_id,
                    'interaction_count': db_profile['interaction_count'],
                    'preferred_genres': db_profile['preferred_genres'],
                    'preferred_price_range': db_profile['preferred_price_range'],
                    'preferred_rating_range': db_profile['preferred_rating_range'],
                    'liked_games': db_profile['liked_games'],
                    'disliked_games': db_profile['disliked_games'],
                    'search_history': db_profile['search_history'],
                    'session_start': datetime.now(),
                    'last_interaction': datetime.now()
                }
                return profile
        
        # Fallback to in-memory profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                'user_id': user_id,
                'interaction_count': 0,
                'preferred_genres': {},
                'preferred_price_range': None,
                'preferred_rating_range': None,
                'liked_games': [],
                'disliked_games': [],
                'search_history': [],
                'session_start': datetime.now(),
                'last_interaction': datetime.now()
            }
        
        return self.user_profiles[user_id]
    
    def _update_user_profile(self, user_profile, query):
        """Update user profile based on current query"""
        user_profile['search_history'].append({
            'query': query,
            'timestamp': datetime.now()
        })
        user_profile['last_interaction'] = datetime.now()
    
    def _record_interaction(self, user_profile, query, recommendations):
        """Record user interaction for learning"""
        user_profile['interaction_count'] += 1
        
        # Extract preferences from query
        query_lower = query.lower()
        
        # Update genre preferences based on query
        for genre in ['action', 'adventure', 'strategy', 'rpg', 'simulation', 'puzzle', 'indie', 'casual']:
            if genre in query_lower:
                user_profile['preferred_genres'][genre] = user_profile['preferred_genres'].get(genre, 0) + 1
    
    def _get_cold_start_recommendations(self, query, user_profile, num_recommendations):
        """Intelligent cold start recommendations for new users"""
        logger.info(f"🧠 Cold start recommendations for query: {query}")
        
        # 1. Analyze query for intent
        query_analysis = self._analyze_query_intent(query)
        
        # 2. Get popular games in relevant categories
        popular_games = self._get_popular_games_by_intent(query_analysis, num_recommendations)
        
        # 3. Add diversity to prevent filter bubbles
        diverse_games = self._add_diversity_to_recommendations(popular_games, num_recommendations)
        
        # 4. Add explanation for why these games were recommended
        for game in diverse_games:
            game['recommendation_reason'] = self._generate_cold_start_explanation(game, query_analysis)
        
        return diverse_games
    
    def _get_hybrid_recommendations(self, query, user_profile, num_recommendations):
        """Hybrid recommendations for users with some data"""
        logger.info(f"🧠 Hybrid recommendations for query: {query}")
        
        # Combine content-based and collaborative filtering
        content_based = self._get_content_based_recommendations(query, user_profile, num_recommendations // 2)
        collaborative = self._get_collaborative_recommendations(user_profile, num_recommendations // 2)
        
        # Merge and deduplicate
        all_recommendations = content_based + collaborative
        unique_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        # Add explanation
        for game in unique_recommendations:
            game['recommendation_reason'] = self._generate_hybrid_explanation(game, user_profile)
        
        return unique_recommendations[:num_recommendations]
    
    def _get_personalized_recommendations(self, query, user_profile, num_recommendations):
        """Fully personalized recommendations for mature profiles"""
        logger.info(f"🧠 Personalized recommendations for query: {query}")
        
        # Use fine-tuned model with user context
        personalized = self._get_fine_tuned_recommendations(query, user_profile, num_recommendations)
        
        # Add explanation
        for game in personalized:
            game['recommendation_reason'] = self._generate_personalized_explanation(game, user_profile)
        
        return personalized
    
    def _analyze_query_intent(self, query):
        """Analyze user query to understand intent"""
        query_lower = query.lower()
        
        intent = {
            'genres': [],
            'price_preference': None,
            'rating_preference': None,
            'mood': None,
            'complexity': None
        }
        
        # Genre detection
        genre_keywords = {
            'action': ['action', 'shooter', 'fighting', 'racing'],
            'adventure': ['adventure', 'exploration', 'story', 'narrative'],
            'strategy': ['strategy', 'tactical', 'management', 'simulation'],
            'rpg': ['rpg', 'roleplay', 'character', 'level'],
            'puzzle': ['puzzle', 'brain', 'logic', 'mind'],
            'indie': ['indie', 'independent', 'small', 'creative'],
            'casual': ['casual', 'relaxing', 'easy', 'simple']
        }
        
        for genre, keywords in genre_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                intent['genres'].append(genre)
        
        # Price preference
        if any(word in query_lower for word in ['free', 'cheap', 'budget']):
            intent['price_preference'] = 'low'
        elif any(word in query_lower for word in ['premium', 'expensive', 'high-end']):
            intent['price_preference'] = 'high'
        
        # Rating preference
        if any(word in query_lower for word in ['good', 'highly rated', 'popular']):
            intent['rating_preference'] = 'high'
        
        # Mood detection
        if any(word in query_lower for word in ['relaxing', 'calm', 'chill']):
            intent['mood'] = 'relaxing'
        elif any(word in query_lower for word in ['exciting', 'intense', 'thrilling']):
            intent['mood'] = 'exciting'
        
        return intent
    
    def _get_popular_games_by_intent(self, intent, num_recommendations):
        """Get popular games based on analyzed intent"""
        if self.df.empty:
            return []
        
        # Start with all games
        games = self.df.copy()
        
        # Filter by detected genres
        if intent['genres']:
            genre_mask = games['Genres'].str.lower().str.contains('|'.join(intent['genres']), na=False)
            games = games[genre_mask]
        
        # Filter by price preference
        if intent['price_preference'] == 'low':
            games = games[games['Price'] <= 10]
        elif intent['price_preference'] == 'high':
            games = games[games['Price'] > 20]
        
        # Filter by rating preference
        if intent['rating_preference'] == 'high':
            games = games[games['Review_Score'] >= 0.7]
        
        # Sort by popularity (review score * total reviews)
        games['popularity_score'] = games['Review_Score'] * games['Total_Reviews']
        games = games.sort_values('popularity_score', ascending=False)
        
        # Convert to recommendation format
        recommendations = []
        for _, game in games.head(num_recommendations).iterrows():
            recommendations.append({
                'name': str(game.get('Name', 'Unknown Game')),
                'app_id': int(game.get('AppID', 0)),
                'genres': str(game.get('Genres', 'No genres')),
                'price': float(game.get('Price', 0)) if pd.notna(game.get('Price')) else 0,
                'review_score': float(game.get('Review_Score', 0)) if pd.notna(game.get('Review_Score')) else 0,
                'total_reviews': int(game.get('Total_Reviews', 0)) if pd.notna(game.get('Total_Reviews')) else 0,
                'description': str(game.get('About the game', 'No description available')),
                'header_image': str(game.get('Header image', '')),
                'platforms': str(game.get('Platforms', 'Unknown')),
                'release_date': str(game.get('Release date', 'Unknown')),
                'developer': str(game.get('Developer', 'Unknown')),
                'publisher': str(game.get('Publisher', 'Unknown')),
                'tags': str(game.get('Tags', 'No tags')),
                'recommendation_type': 'cold_start'
            })
        
        return recommendations
    
    def _add_diversity_to_recommendations(self, recommendations, num_recommendations):
        """Add diversity to prevent filter bubbles"""
        if len(recommendations) <= num_recommendations:
            return recommendations
        
        # Ensure we have games from different genres
        diverse_recommendations = []
        used_genres = set()
        
        # First pass: one game per genre
        for game in recommendations:
            if len(diverse_recommendations) >= num_recommendations:
                break
            game_genres = str(game.get('genres', '')).lower()
            if not any(genre in used_genres for genre in game_genres.split(',')):
                diverse_recommendations.append(game)
                used_genres.update(game_genres.split(','))
        
        # Second pass: fill remaining slots
        for game in recommendations:
            if len(diverse_recommendations) >= num_recommendations:
                break
            if game not in diverse_recommendations:
                diverse_recommendations.append(game)
        
        return diverse_recommendations
    
    def _get_content_based_recommendations(self, query, user_profile, num_recommendations):
        """Content-based recommendations using game features"""
        # Use the existing similarity method but with user preferences
        return self._get_similarity_recommendations(query, num_recommendations)
    
    def _get_collaborative_recommendations(self, user_profile, num_recommendations):
        """Collaborative filtering recommendations"""
        # For now, return popular games. In production, this would use user similarity
        return self._get_popular_games_by_intent({'genres': [], 'price_preference': None, 'rating_preference': 'high'}, num_recommendations)
    
    def _get_fine_tuned_recommendations(self, query, user_profile, num_recommendations):
        """Use fine-tuned model for recommendations"""
        # For now, use similarity. In production, this would use the fine-tuned model
        return self._get_similarity_recommendations(query, num_recommendations)
    
    def _deduplicate_recommendations(self, recommendations):
        """Remove duplicate recommendations"""
        seen = set()
        unique = []
        for rec in recommendations:
            if rec['app_id'] not in seen:
                seen.add(rec['app_id'])
                unique.append(rec)
        return unique
    
    def _generate_cold_start_explanation(self, game, intent):
        """Generate explanation for cold start recommendations"""
        reasons = []
        
        if intent['genres']:
            reasons.append(f"Matches your interest in {', '.join(intent['genres'])} games")
        
        if intent['price_preference'] == 'low':
            reasons.append("Fits your budget preference")
        elif intent['price_preference'] == 'high':
            reasons.append("Premium quality game")
        
        if intent['rating_preference'] == 'high':
            reasons.append("Highly rated by the community")
        
        if not reasons:
            reasons.append("Popular and well-reviewed game")
        
        return " | ".join(reasons)
    
    def _generate_hybrid_explanation(self, game, user_profile):
        """Generate explanation for hybrid recommendations"""
        reasons = []
        
        if user_profile['preferred_genres']:
            top_genre = max(user_profile['preferred_genres'], key=user_profile['preferred_genres'].get)
            if top_genre.lower() in str(game.get('genres', '')).lower():
                reasons.append(f"Matches your interest in {top_genre} games")
        
        if game.get('recommendation_type') == 'cold_start':
            reasons.append("Popular choice for new users")
        else:
            reasons.append("Based on similar users' preferences")
        
        return " | ".join(reasons) if reasons else "Recommended based on your preferences"
    
    def _generate_personalized_explanation(self, game, user_profile):
        """Generate explanation for personalized recommendations"""
        reasons = []
        
        if user_profile['preferred_genres']:
            top_genre = max(user_profile['preferred_genres'], key=user_profile['preferred_genres'].get)
            if top_genre.lower() in str(game.get('genres', '')).lower():
                reasons.append(f"Perfect match for your {top_genre} preference")
        
        if user_profile['interaction_count'] > 20:
            reasons.append("Tailored to your gaming history")
        
        return " | ".join(reasons) if reasons else "Personally recommended for you"
    
    def _get_similarity_recommendations(self, query, num_recommendations=5):
        """Get similarity-based recommendations from actual dataset"""
        if self.df.empty:
            return []
        
        try:
            # Enhanced search across multiple fields
            query_lower = query.lower()
            
            # Search in name, genres, tags, and description
            name_mask = self.df['Name'].str.lower().str.contains(query_lower, na=False)
            genre_mask = self.df['Genres'].str.lower().str.contains(query_lower, na=False)
            tag_mask = self.df['Tags'].str.lower().str.contains(query_lower, na=False)
            desc_mask = self.df['About the game'].str.lower().str.contains(query_lower, na=False)
            
            # Combine all search criteria
            combined_mask = name_mask | genre_mask | tag_mask | desc_mask
            
            # Filter and sort by quality
            matching_games = self.df[combined_mask].copy()
            
            if len(matching_games) == 0:
                return []
            
            # Sort by review score and total reviews for better quality
            matching_games = matching_games.sort_values(
                ['Review_Score', 'Total_Reviews'], 
                ascending=[False, False]
            ).head(num_recommendations)
            
            recommendations = []
            for _, game in matching_games.iterrows():
                # Safely extract data with fallbacks
                recommendations.append({
                    'name': str(game.get('Name', 'Unknown Game')),
                    'app_id': int(game.get('AppID', 0)),
                    'genres': str(game.get('Genres', 'No genres')),
                    'price': float(game.get('Price', 0)) if pd.notna(game.get('Price')) else 0,
                    'review_score': float(game.get('Review_Score', 0)) if pd.notna(game.get('Review_Score')) else 0,
                    'total_reviews': int(game.get('Total_Reviews', 0)) if pd.notna(game.get('Total_Reviews')) else 0,
                    'description': str(game.get('About the game', 'No description available'))[:200] + '...' if pd.notna(game.get('About the game')) else 'No description available',
                    'header_image': str(game.get('Header image', '')) if pd.notna(game.get('Header image')) else '',
                    'release_date': str(game.get('Release date', 'Unknown')) if pd.notna(game.get('Release date')) else 'Unknown',
                    'platforms': {
                        'windows': bool(game.get('Windows', False)),
                        'mac': bool(game.get('Mac', False)),
                        'linux': bool(game.get('Linux', False))
                    },
                    'developers': str(game.get('Developers', '')) if pd.notna(game.get('Developers')) else '',
                    'publishers': str(game.get('Publishers', '')) if pd.notna(game.get('Publishers')) else ''
                })
            
            return recommendations
            
        except Exception as e:
            logger.error(f"❌ Error in similarity search: {e}")
            return []
    
    def get_genre_stats(self):
        """Get genre distribution statistics"""
        if self.df.empty:
            return {}
        
        # Count games by genre
        genre_counts = {}
        for genres in self.df['Genres'].dropna():
            if isinstance(genres, str):
                for genre in genres.split(';'):
                    genre = genre.strip()
                    if genre:
                        genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Sort by count
        sorted_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_genres[:20])  # Top 20 genres
    
    def get_price_distribution(self):
        """Get price distribution data"""
        if self.df.empty:
            return {}
        
        # Filter out free games and extreme outliers
        price_data = self.df[
            (self.df['Price'] > 0) & 
            (self.df['Price'] < 100) & 
            (self.df['Price'].notna())
        ]['Price']
        
        if price_data.empty:
            return {}
        
        # Create price bins
        bins = [0, 5, 10, 20, 30, 50, 100]
        labels = ['$0-5', '$5-10', '$10-20', '$20-30', '$30-50', '$50+']
        
        price_distribution = pd.cut(price_data, bins=bins, labels=labels, include_lowest=True)
        price_counts = price_distribution.value_counts().to_dict()
        
        return price_counts
    
    def get_rating_distribution(self):
        """Get review score distribution"""
        if self.df.empty:
            return {}
        
        # Filter valid review scores
        rating_data = self.df[
            (self.df['Review_Score'] >= 0) & 
            (self.df['Review_Score'] <= 1) & 
            (self.df['Review_Score'].notna())
        ]['Review_Score']
        
        if rating_data.empty:
            return {}
        
        # Create rating bins
        bins = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        labels = ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
        
        rating_distribution = pd.cut(rating_data, bins=bins, labels=labels, include_lowest=True)
        rating_counts = rating_distribution.value_counts().to_dict()
        
        return rating_counts

# Initialize dashboard
dashboard = GameMatchDashboard()

# Authentication decorator
def login_required(f):
    """Decorator to require login for protected routes"""
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    decorated_function.__name__ = f.__name__
    return decorated_function

@app.route('/')
def index():
    """Main dashboard page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('dashboard.html', user=session.get('user_info', {}))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        
        if not email or not password:
            flash('Please fill in all fields', 'error')
            return render_template('login.html')
        
        success, result = dashboard.auth.authenticate_user(email, password)
        
        if success:
            session['user_id'] = result['user_id']
            session['user_info'] = {
                'username': result['username'],
                'email': result['email']
            }
            session.permanent = True
            flash(f'Welcome back, {result["username"]}!', 'success')
            return redirect(url_for('index'))
        else:
            flash(result, 'error')
    
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration page"""
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        username = request.form.get('username', '').strip()
        
        if not all([email, password, username]):
            flash('Please fill in all fields', 'error')
            return render_template('register.html')
        
        if len(password) < 6:
            flash('Password must be at least 6 characters long', 'error')
            return render_template('register.html')
        
        success, message = dashboard.auth.register_user(email, password, username)
        
        if success:
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))
        else:
            flash(message, 'error')
    
    return render_template('register.html')

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    flash('You have been logged out', 'info')
    return redirect(url_for('login'))

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'games_loaded': len(dashboard.df) if dashboard.df is not None else 0
    })

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'status': 'operational',
        'version': '1.0.0',
        'games_available': len(dashboard.df) if dashboard.df is not None else 0,
        'last_updated': datetime.now().isoformat()
    })

@app.route('/api/recommendations')
def api_recommendations():
    """API endpoint for intelligent game recommendations with proper error handling"""
    try:
        query = request.args.get('query', '').strip()
        # Use authenticated user ID or fallback to anonymous
        user_id = session.get('user_id', 'anonymous')
        num_recommendations = min(int(request.args.get('limit', 5)), 50)  # Cap at 50
        
        if not query:
            return jsonify({
                'error': 'Query parameter is required',
                'message': 'Please provide a search query'
            }), 400
        
        if dashboard.df is None or dashboard.df.empty:
            return jsonify({
                'error': 'Dataset not available',
                'message': 'Game dataset is not loaded. Please try again later.'
            }), 503
        
        # Get intelligent recommendations
        recommendations = dashboard.get_game_recommendations(query, user_id, num_recommendations)
        
        # Get user profile info for response
        user_profile = dashboard._get_or_create_user_profile(user_id)
        
        # Save user profile to database if authenticated
        if user_id != 'anonymous':
            dashboard.auth.update_user_preferences(user_id, user_profile)
        
        # Save recommendation for analysis
        if user_id != 'anonymous':
            dashboard.auth.save_recommendation(
                user_id, 
                query, 
                [{'app_id': game['app_id'], 'name': game['name']} for game in recommendations],
                'cold_start' if user_profile['interaction_count'] < 3 else 
                'hybrid' if user_profile['interaction_count'] < 10 else 'personalized'
            )
        
        return jsonify({
            'query': query,
            'count': len(recommendations),
            'recommendations': recommendations,
            'user_profile': {
                'interaction_count': user_profile['interaction_count'],
                'preferred_genres': user_profile['preferred_genres'],
                'recommendation_type': 'cold_start' if user_profile['interaction_count'] < 3 else 
                                     'hybrid' if user_profile['interaction_count'] < 10 else 'personalized'
            }
        })
        
    except ValueError as e:
        return jsonify({
            'error': 'Invalid parameter',
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"Error in recommendations API: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'An error occurred while processing your request'
        }), 500

@app.route('/api/stats')
def api_stats():
    """API endpoint for dashboard statistics"""
    try:
        if dashboard.df is None or dashboard.df.empty:
            return jsonify({
                'error': 'Dataset not available',
                'genre_stats': {},
                'price_distribution': {},
                'rating_distribution': {},
                'total_games': 0
            }), 503
        
        genre_stats = dashboard.get_genre_stats()
        price_distribution = dashboard.get_price_distribution()
        rating_distribution = dashboard.get_rating_distribution()
        
        return jsonify({
            'genre_stats': genre_stats,
            'price_distribution': price_distribution,
            'rating_distribution': rating_distribution,
            'total_games': len(dashboard.df)
        })
    except Exception as e:
        logger.error(f"Error generating stats: {e}")
        return jsonify({
            'error': 'Internal server error',
            'genre_stats': {},
            'price_distribution': {},
            'rating_distribution': {},
            'total_games': 0
        }), 500

@app.route('/api/games/browse')
def api_browse_games():
    """API endpoint for browsing games by genre, price, rating, etc."""
    try:
        if dashboard.df is None or dashboard.df.empty:
            return jsonify({'error': 'Dataset not available'}), 503
        
        # Get filter parameters
        genre = request.args.get('genre', '')
        price_range = request.args.get('price_range', '')
        rating_range = request.args.get('rating_range', '')
        limit = min(int(request.args.get('limit', 20)), 100)
        page = int(request.args.get('page', 1))
        
        # Start with full dataset
        filtered_df = dashboard.df.copy()
        
        # Apply genre filter
        if genre and genre != 'all':
            filtered_df = filtered_df[filtered_df['Genres'].str.contains(genre, case=False, na=False)]
        
        # Apply price filter
        if price_range:
            if price_range == 'free':
                filtered_df = filtered_df[filtered_df['Price'] == 0]
            elif price_range == 'under10':
                filtered_df = filtered_df[filtered_df['Price'] < 10]
            elif price_range == '10to30':
                filtered_df = filtered_df[(filtered_df['Price'] >= 10) & (filtered_df['Price'] <= 30)]
            elif price_range == 'over30':
                filtered_df = filtered_df[filtered_df['Price'] > 30]
        
        # Apply rating filter
        if rating_range:
            if rating_range == 'high':
                filtered_df = filtered_df[filtered_df['Review_Score'] >= 0.8]
            elif rating_range == 'good':
                filtered_df = filtered_df[filtered_df['Review_Score'] >= 0.6]
            elif rating_range == 'mixed':
                filtered_df = filtered_df[filtered_df['Review_Score'] >= 0.4]
        
        # Sort by popularity (total reviews) and rating
        filtered_df = filtered_df.sort_values(['Total_Reviews', 'Review_Score'], ascending=[False, False])
        
        # Pagination
        start_idx = (page - 1) * limit
        end_idx = start_idx + limit
        page_df = filtered_df.iloc[start_idx:end_idx]
        
        # Format games for response
        games = []
        for _, game in page_df.iterrows():
            games.append({
                'app_id': int(game['AppID']),
                'name': game['Name'],
                'genres': game['Genres'],
                'price': float(game['Price']) if pd.notna(game['Price']) else 0.0,
                'review_score': float(game['Review_Score']) if pd.notna(game['Review_Score']) else 0.0,
                'total_reviews': int(game['Total_Reviews']) if pd.notna(game['Total_Reviews']) else 0,
                'description': game['About the game'][:200] + '...' if pd.notna(game['About the game']) and len(str(game['About the game'])) > 200 else str(game['About the game']),
                'header_image': dashboard.clean_image_url(game['Header image']) if pd.notna(game['Header image']) else '',
                'release_date': str(game['Release date']) if pd.notna(game['Release date']) else '',
                'developer': game['Developers'] if pd.notna(game['Developers']) else 'Unknown',
                'publisher': game['Publishers'] if pd.notna(game['Publishers']) else 'Unknown',
                'tags': game['Tags'] if pd.notna(game['Tags']) else ''
            })
        
        return jsonify({
            'games': games,
            'total_count': len(filtered_df),
            'page': page,
            'limit': limit,
            'total_pages': (len(filtered_df) + limit - 1) // limit,
            'filters_applied': {
                'genre': genre,
                'price_range': price_range,
                'rating_range': rating_range
            }
        })
        
    except Exception as e:
        logger.error(f"Error browsing games: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/game/<int:app_id>')
@login_required
def api_game_details(app_id):
    """API endpoint for individual game details"""
    try:
        if dashboard.df is None or dashboard.df.empty:
            return jsonify({'error': 'Dataset not available'}), 503
        
        game = dashboard.df[dashboard.df['AppID'] == app_id]
        
        if game.empty:
            return jsonify({'error': 'Game not found'}), 404
        
        game_data = game.iloc[0]
        
        # Track game view
        user_id = session.get('user_id', 'anonymous')
        if user_id != 'anonymous':
            dashboard.auth.track_game_view(user_id, app_id)
        
        return jsonify({
            'name': str(game_data.get('Name', 'Unknown Game')),
            'app_id': int(game_data.get('AppID', 0)),
            'genres': str(game_data.get('Genres', 'No genres')),
            'price': float(game_data.get('Price', 0)) if pd.notna(game_data.get('Price')) else 0,
            'review_score': float(game_data.get('Review_Score', 0)) if pd.notna(game_data.get('Review_Score')) else 0,
            'total_reviews': int(game_data.get('Total_Reviews', 0)) if pd.notna(game_data.get('Total_Reviews')) else 0,
            'description': str(game_data.get('About the game', 'No description available')),
            'header_image': str(game_data.get('Header image', '')),
            'release_date': str(game_data.get('Release date', 'Unknown')),
            'platforms': str(game_data.get('Platforms', 'Unknown')),
            'developers': str(game_data.get('Developer', 'Unknown')),
            'publishers': str(game_data.get('Publisher', 'Unknown')),
            'tags': str(game_data.get('Tags', 'No tags'))
        })
        
    except Exception as e:
        logger.error(f"Error getting game details: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/game/<int:game_id>/interact', methods=['POST'])
@login_required
def track_game_interaction(game_id):
    """Track user interaction with a game"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if user_id == 'anonymous':
            return jsonify({'error': 'Authentication required'}), 401
        
        data = request.get_json()
        interaction_type = data.get('type', 'view')
        interaction_data = data.get('data', {})
        
        # Track the interaction
        success = dashboard.auth.track_game_interaction(
            user_id, 
            game_id, 
            interaction_type, 
            interaction_data
        )
        
        if success:
            return jsonify({'status': 'success', 'message': 'Interaction tracked'})
        else:
            return jsonify({'error': 'Failed to track interaction'}), 500
            
    except Exception as e:
        logger.error(f"Error tracking game interaction: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/user/interactions')
@login_required
def get_user_interactions():
    """Get user's game interactions"""
    try:
        user_id = session.get('user_id', 'anonymous')
        if user_id == 'anonymous':
            return jsonify({'error': 'Authentication required'}), 401
        
        interactions = dashboard.auth.get_user_interactions(user_id)
        return jsonify({'interactions': interactions})
        
    except Exception as e:
        logger.error(f"Error getting user interactions: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🎮 Starting GameMatch Visual Dashboard...")
    print("🌐 Dashboard will be available at: http://localhost:5000")
    print("📊 Features:")
    print("   • Interactive game recommendations")
    print("   • Visual statistics and charts")
    print("   • Game details and images")
    print("   • Real-time search and filtering")
    print("\n🔧 Configuration:")
    print(f"   • Environment: {os.getenv('ENVIRONMENT', 'development')}")
    print(f"   • Debug mode: {os.getenv('DEBUG_MODE', 'false')}")
    print(f"   • Database: {os.getenv('MSSQL_SERVER', 'localhost')}")
    
    # Use environment-based configuration for debug mode
    debug_mode = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
    port = int(os.getenv('FLASK_PORT', '5000'))
    host = os.getenv('FLASK_HOST', '0.0.0.0' if debug_mode else '127.0.0.1')
    
    # Add simple test route
    @app.route('/test')
    def test_page():
        """Simple test page"""
        with open('test_simple.html', 'r') as f:
            return f.read()
    
    logger.info(f"Starting Flask app on {host}:{port} (debug={debug_mode})")
    app.run(debug=debug_mode, host=host, port=port)