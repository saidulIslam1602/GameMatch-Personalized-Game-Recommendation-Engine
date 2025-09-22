#!/usr/bin/env python3
"""
GameMatch MSSQL Authentication System
Microsoft SQL Server-based user authentication and management
"""

import pyodbc
import hashlib
import secrets
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any

logger = logging.getLogger(__name__)

class MSSQLUserAuth:
    """Microsoft SQL Server-based user authentication and management system"""
    
    def __init__(self, server=None, port=None, database=None, 
                 username=None, password=None):
        # Use environment variables with secure defaults
        self.server = server or os.getenv('MSSQL_SERVER', 'localhost')
        self.port = port or int(os.getenv('MSSQL_PORT', '1433'))
        self.database = database or os.getenv('MSSQL_DATABASE', 'gamematch')
        self.username = username or os.getenv('MSSQL_USERNAME', 'sa')
        self.password = password or os.getenv('MSSQL_PASSWORD')
        
        if not self.password:
            raise ValueError("MSSQL_PASSWORD environment variable is required")
        self.connection_string = self._build_connection_string()
        self.init_database()
    
    def _build_connection_string(self) -> str:
        """Build MSSQL connection string"""
        return f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.server},{self.port};DATABASE={self.database};UID={self.username};PWD={self.password};TrustServerCertificate=yes"
    
    def _get_connection(self):
        """Get MSSQL database connection"""
        try:
            return pyodbc.connect(self.connection_string)
        except Exception as e:
            logger.error(f"❌ Failed to connect to MSSQL: {e}")
            raise
    
    def init_database(self):
        """Initialize MSSQL database tables"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='users' AND xtype='U')
                CREATE TABLE users (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    email NVARCHAR(255) UNIQUE NOT NULL,
                    password_hash NVARCHAR(500) NOT NULL,
                    username NVARCHAR(255) NOT NULL,
                    created_at DATETIME2 DEFAULT GETDATE(),
                    last_login DATETIME2,
                    is_active BIT DEFAULT 1
                )
            ''')
            
            # Create user preferences table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_preferences' AND xtype='U')
                CREATE TABLE user_preferences (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NOT NULL,
                    preferred_genres NVARCHAR(MAX),
                    preferred_price_range NVARCHAR(100),
                    preferred_rating_range NVARCHAR(100),
                    liked_games NVARCHAR(MAX),
                    disliked_games NVARCHAR(MAX),
                    search_history NVARCHAR(MAX),
                    interaction_count INT DEFAULT 0,
                    last_updated DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create game interactions table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='game_interactions' AND xtype='U')
                CREATE TABLE game_interactions (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NOT NULL,
                    game_id INT NOT NULL,
                    interaction_type NVARCHAR(100) NOT NULL,
                    interaction_data NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create game views table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='game_views' AND xtype='U')
                CREATE TABLE game_views (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NOT NULL,
                    game_id INT NOT NULL,
                    view_duration INT DEFAULT 0,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create user recommendations table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='user_recommendations' AND xtype='U')
                CREATE TABLE user_recommendations (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    user_id INT NOT NULL,
                    query NVARCHAR(500) NOT NULL,
                    recommended_games NVARCHAR(MAX) NOT NULL,
                    recommendation_type NVARCHAR(100) NOT NULL,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    FOREIGN KEY (user_id) REFERENCES users (id)
                )
            ''')
            
            # Create performance metrics table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='performance_metrics' AND xtype='U')
                CREATE TABLE performance_metrics (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    metric_name NVARCHAR(100) NOT NULL,
                    metric_value FLOAT NOT NULL,
                    user_id INT,
                    session_id NVARCHAR(100),
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    metadata NVARCHAR(MAX),
                    INDEX IX_performance_metrics_timestamp (timestamp),
                    INDEX IX_performance_metrics_metric_name (metric_name)
                )
            ''')
            
            # Create A/B testing experiments table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ab_experiments' AND xtype='U')
                CREATE TABLE ab_experiments (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    experiment_id NVARCHAR(100) UNIQUE NOT NULL,
                    experiment_name NVARCHAR(255) NOT NULL,
                    description NVARCHAR(MAX),
                    variants NVARCHAR(MAX) NOT NULL,
                    traffic_allocation NVARCHAR(MAX) NOT NULL,
                    success_metrics NVARCHAR(MAX) NOT NULL,
                    status NVARCHAR(50) DEFAULT 'created',
                    start_date DATETIME2,
                    end_date DATETIME2,
                    created_at DATETIME2 DEFAULT GETDATE()
                )
            ''')
            
            # Create A/B testing user assignments table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ab_user_assignments' AND xtype='U')
                CREATE TABLE ab_user_assignments (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    experiment_id NVARCHAR(100) NOT NULL,
                    user_id NVARCHAR(100) NOT NULL,
                    variant NVARCHAR(100) NOT NULL,
                    assigned_at DATETIME2 DEFAULT GETDATE(),
                    UNIQUE (experiment_id, user_id),
                    INDEX IX_ab_assignments_experiment_user (experiment_id, user_id)
                )
            ''')
            
            # Create A/B testing interactions table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='ab_interactions' AND xtype='U')
                CREATE TABLE ab_interactions (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    experiment_id NVARCHAR(100) NOT NULL,
                    user_id NVARCHAR(100) NOT NULL,
                    variant NVARCHAR(100) NOT NULL,
                    success_metrics NVARCHAR(MAX),
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    INDEX IX_ab_interactions_experiment (experiment_id),
                    INDEX IX_ab_interactions_timestamp (timestamp)
                )
            ''')
            
            # Create system health monitoring table
            cursor.execute('''
                IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='system_health' AND xtype='U')
                CREATE TABLE system_health (
                    id INT IDENTITY(1,1) PRIMARY KEY,
                    component_name NVARCHAR(100) NOT NULL,
                    status NVARCHAR(50) NOT NULL,
                    response_time_ms FLOAT,
                    error_count INT DEFAULT 0,
                    cpu_usage FLOAT,
                    memory_usage FLOAT,
                    timestamp DATETIME2 DEFAULT GETDATE(),
                    metadata NVARCHAR(MAX),
                    INDEX IX_system_health_timestamp (timestamp),
                    INDEX IX_system_health_component (component_name)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("✅ MSSQL production database initialized with performance tracking")
            
        except Exception as e:
            logger.error(f"❌ Error initializing MSSQL database: {e}")
            # Fallback: create database if it doesn't exist
            try:
                master_conn_string = f"DRIVER={{ODBC Driver 18 for SQL Server}};SERVER={self.server},{self.port};DATABASE=master;UID={self.username};PWD={self.password};TrustServerCertificate=yes"
                master_conn = pyodbc.connect(master_conn_string)
                master_conn.autocommit = True
                master_cursor = master_conn.cursor()
                
                # Check if database exists first
                master_cursor.execute(f"SELECT COUNT(*) FROM sys.databases WHERE name = '{self.database}'")
                db_exists = master_cursor.fetchone()[0]
                
                if not db_exists:
                    master_cursor.execute(f"CREATE DATABASE [{self.database}]")
                    logger.info(f"✅ Created database '{self.database}'")
                else:
                    logger.info(f"✅ Database '{self.database}' already exists")
                
                master_conn.close()
                
                # Wait a moment for database to be ready
                import time
                time.sleep(2)
                
                # Retry initialization
                self.init_database()
            except Exception as create_error:
                logger.error(f"❌ Failed to create database: {create_error}")
                raise
    
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
            conn = self._get_connection()
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
            
            # Get the user ID
            cursor.execute("SELECT @@IDENTITY")
            user_id = cursor.fetchone()[0]
            
            # Create default preferences
            cursor.execute(
                "INSERT INTO user_preferences (user_id) VALUES (?)",
                (user_id,)
            )
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ User registered in MSSQL: {email}")
            return True, "Registration successful"
            
        except Exception as e:
            logger.error(f"❌ MSSQL registration error: {e}")
            return False, "Registration failed"
    
    def authenticate_user(self, email, password):
        """Authenticate user login"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute(
                "SELECT id, password_hash, username FROM users WHERE email = ? AND is_active = 1",
                (email,)
            )
            user = cursor.fetchone()
            
            if user and self.verify_password(password, user[1]):
                # Update last login
                cursor.execute(
                    "UPDATE users SET last_login = GETDATE() WHERE id = ?",
                    (user[0],)
                )
                conn.commit()
                conn.close()
                
                logger.info(f"✅ User authenticated in MSSQL: {email}")
                return True, {"user_id": user[0], "username": user[2], "email": email}
            else:
                conn.close()
                return False, "Invalid credentials"
                
        except Exception as e:
            logger.error(f"❌ MSSQL authentication error: {e}")
            return False, "Authentication failed"
    
    def get_user_preferences(self, user_id):
        """Get user preferences from MSSQL database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT preferred_genres, preferred_price_range, preferred_rating_range,
                       liked_games, disliked_games, search_history, interaction_count
                FROM user_preferences
                WHERE user_id = ?
            ''', (user_id,))
            
            prefs = cursor.fetchone()
            conn.close()
            
            if prefs:
                return {
                    'preferred_genres': json.loads(prefs[0]) if prefs[0] else {},
                    'preferred_price_range': prefs[1],
                    'preferred_rating_range': prefs[2],
                    'liked_games': json.loads(prefs[3]) if prefs[3] else [],
                    'disliked_games': json.loads(prefs[4]) if prefs[4] else [],
                    'search_history': json.loads(prefs[5]) if prefs[5] else [],
                    'interaction_count': prefs[6] or 0
                }
            else:
                return {
                    'preferred_genres': {},
                    'preferred_price_range': None,
                    'preferred_rating_range': None,
                    'liked_games': [],
                    'disliked_games': [],
                    'search_history': [],
                    'interaction_count': 0
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting user preferences from MSSQL: {e}")
            return {
                'preferred_genres': {},
                'preferred_price_range': None,
                'preferred_rating_range': None,
                'liked_games': [],
                'disliked_games': [],
                'search_history': [],
                'interaction_count': 0
            }
    
    def update_user_preferences(self, user_id, preferences):
        """Update user preferences in MSSQL database"""
        try:
            conn = self._get_connection()
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
                    last_updated = GETDATE()
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
            logger.error(f"❌ Error updating user preferences in MSSQL: {e}")
            return False
    
    def save_recommendation(self, user_id, query, recommended_games, recommendation_type):
        """Save recommendation for analysis in MSSQL"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO user_recommendations (user_id, query, recommended_games, recommendation_type)
                VALUES (?, ?, ?, ?)
            ''', (user_id, query, json.dumps(recommended_games), recommendation_type))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving recommendation to MSSQL: {e}")
            return False

    def track_game_interaction(self, user_id, game_id, interaction_type):
        """Track user's game interaction"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO user_interactions (user_id, game_id, interaction_type, timestamp)
                VALUES (?, ?, ?, GETDATE())
            """, (user_id, game_id, interaction_type))
            
            conn.commit()
            conn.close()
            logger.info(f"✅ Tracked interaction: user {user_id}, game {game_id}, type {interaction_type}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error tracking game interaction: {e}")
            return False

    def get_user_interactions(self, user_id):
        """Get user's game interactions"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT game_id, interaction_type, timestamp 
                FROM user_interactions 
                WHERE user_id = ? 
                ORDER BY timestamp DESC
            """, (user_id,))
            
            interactions = []
            for row in cursor.fetchall():
                interactions.append({
                    'game_id': row[0],
                    'interaction_type': row[1],
                    'timestamp': row[2]
                })
            
            conn.close()
            return interactions
            
        except Exception as e:
            logger.error(f"❌ Error getting user interactions: {e}")
            return []
    
    def save_performance_metric(self, metric_name, metric_value, user_id=None, session_id=None, metadata=None):
        """Save performance metric to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO performance_metrics (metric_name, metric_value, user_id, session_id, metadata)
                VALUES (?, ?, ?, ?, ?)
            """, (metric_name, metric_value, user_id, session_id, json.dumps(metadata) if metadata else None))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving performance metric: {e}")
            return False
    
    def get_performance_metrics(self, metric_name=None, hours=24):
        """Get performance metrics from database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            if metric_name:
                cursor.execute("""
                    SELECT metric_name, metric_value, timestamp, metadata
                    FROM performance_metrics 
                    WHERE metric_name = ? AND timestamp >= DATEADD(hour, -?, GETDATE())
                    ORDER BY timestamp DESC
                """, (metric_name, hours))
            else:
                cursor.execute("""
                    SELECT metric_name, metric_value, timestamp, metadata
                    FROM performance_metrics 
                    WHERE timestamp >= DATEADD(hour, -?, GETDATE())
                    ORDER BY timestamp DESC
                """, (hours,))
            
            metrics = []
            for row in cursor.fetchall():
                metrics.append({
                    'metric_name': row[0],
                    'metric_value': row[1],
                    'timestamp': row[2],
                    'metadata': json.loads(row[3]) if row[3] else None
                })
            
            conn.close()
            return metrics
            
        except Exception as e:
            logger.error(f"❌ Error getting performance metrics: {e}")
            return []
    
    def save_ab_experiment(self, experiment_id, experiment_name, description, variants, traffic_allocation, success_metrics):
        """Save A/B testing experiment to database"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ab_experiments (experiment_id, experiment_name, description, variants, traffic_allocation, success_metrics)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                experiment_id, 
                experiment_name, 
                description,
                json.dumps(variants),
                json.dumps(traffic_allocation),
                json.dumps(success_metrics)
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving A/B experiment: {e}")
            return False
    
    def assign_user_to_variant(self, experiment_id, user_id, variant):
        """Assign user to A/B test variant"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Check if user already assigned
            cursor.execute("""
                SELECT variant FROM ab_user_assignments 
                WHERE experiment_id = ? AND user_id = ?
            """, (experiment_id, user_id))
            
            existing = cursor.fetchone()
            if existing:
                conn.close()
                return existing[0]
            
            # Assign new variant
            cursor.execute("""
                INSERT INTO ab_user_assignments (experiment_id, user_id, variant)
                VALUES (?, ?, ?)
            """, (experiment_id, user_id, variant))
            
            conn.commit()
            conn.close()
            return variant
            
        except Exception as e:
            logger.error(f"❌ Error assigning user to variant: {e}")
            return "control"  # Default fallback
    
    def save_ab_interaction(self, experiment_id, user_id, variant, success_metrics):
        """Save A/B testing interaction"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO ab_interactions (experiment_id, user_id, variant, success_metrics)
                VALUES (?, ?, ?, ?)
            """, (experiment_id, user_id, variant, json.dumps(success_metrics)))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving A/B interaction: {e}")
            return False
    
    def get_ab_experiment_results(self, experiment_id):
        """Get A/B testing experiment results"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get experiment details
            cursor.execute("""
                SELECT experiment_name, variants, success_metrics
                FROM ab_experiments 
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            experiment = cursor.fetchone()
            if not experiment:
                conn.close()
                return None
            
            # Get interaction data
            cursor.execute("""
                SELECT variant, success_metrics, timestamp
                FROM ab_interactions 
                WHERE experiment_id = ?
            """, (experiment_id,))
            
            interactions = []
            for row in cursor.fetchall():
                interactions.append({
                    'variant': row[0],
                    'success_metrics': json.loads(row[1]) if row[1] else {},
                    'timestamp': row[2]
                })
            
            conn.close()
            
            return {
                'experiment_name': experiment[0],
                'variants': json.loads(experiment[1]),
                'success_metrics': json.loads(experiment[2]),
                'interactions': interactions
            }
            
        except Exception as e:
            logger.error(f"❌ Error getting A/B experiment results: {e}")
            return None
    
    def save_system_health(self, component_name, status, response_time_ms=None, error_count=0, cpu_usage=None, memory_usage=None, metadata=None):
        """Save system health metrics"""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO system_health (component_name, status, response_time_ms, error_count, cpu_usage, memory_usage, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (component_name, status, response_time_ms, error_count, cpu_usage, memory_usage, json.dumps(metadata) if metadata else None))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving system health: {e}")
            return False