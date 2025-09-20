"""
GameMatch Gaming Ontology and Taxonomy System
Advanced hierarchical classification for games with detailed taxonomies
"""

import json
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import re
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class GameTaxonomy:
    """Structured game taxonomy with hierarchical relationships"""
    primary_genre: str
    sub_genres: List[str]
    mechanics: List[str] 
    themes: List[str]
    target_audience: str
    complexity_level: str
    time_investment: str
    platform_preferences: List[str]
    
class GamingOntologySystem:
    """Advanced gaming ontology with hierarchical classification"""
    
    def __init__(self):
        self.genre_hierarchy = self._build_genre_hierarchy()
        self.mechanics_taxonomy = self._build_mechanics_taxonomy()
        self.theme_ontology = self._build_theme_ontology()
        self.audience_classification = self._build_audience_classification()
        self.complexity_levels = self._build_complexity_levels()
        
    def _build_genre_hierarchy(self) -> Dict:
        """Build comprehensive genre hierarchy"""
        return {
            "Action": {
                "subgenres": ["Beat 'em up", "Fighting", "Hack and slash", "Platform", "Shooter"],
                "mechanics": ["Real-time combat", "Reflexes", "Timing", "Combo systems"],
                "typical_features": ["Fast-paced", "Hand-eye coordination", "Immediate feedback"]
            },
            "Adventure": {
                "subgenres": ["Action-adventure", "Visual novel", "Interactive fiction", "Walking simulator"],
                "mechanics": ["Story progression", "Dialogue choices", "Exploration", "Puzzle solving"],
                "typical_features": ["Narrative-driven", "Character development", "World exploration"]
            },
            "Role-Playing (RPG)": {
                "subgenres": ["Action RPG", "JRPG", "CRPG", "Tactical RPG", "MMORPG"],
                "mechanics": ["Character progression", "Stat management", "Party systems", "Quest systems"],
                "typical_features": ["Character customization", "Story depth", "Long-term progression"]
            },
            "Strategy": {
                "subgenres": ["Real-time strategy (RTS)", "Turn-based strategy (TBS)", "4X", "Tower defense", "Grand strategy"],
                "mechanics": ["Resource management", "Unit control", "Territory control", "Strategic planning"],
                "typical_features": ["Tactical thinking", "Long-term planning", "Complex systems"]
            },
            "Simulation": {
                "subgenres": ["City building", "Life simulation", "Vehicle simulation", "Management simulation"],
                "mechanics": ["System management", "Economy balancing", "Growth mechanics", "Optimization"],
                "typical_features": ["Realistic mechanics", "Creative freedom", "Sandbox elements"]
            },
            "Puzzle": {
                "subgenres": ["Logic puzzle", "Physics puzzle", "Match-3", "Escape room"],
                "mechanics": ["Pattern recognition", "Logical deduction", "Spatial reasoning"],
                "typical_features": ["Mental challenge", "Progressive difficulty", "Eureka moments"]
            },
            "Sports & Racing": {
                "subgenres": ["Arcade racing", "Simulation racing", "Team sports", "Individual sports"],
                "mechanics": ["Physics simulation", "Timing", "Precision control", "Competition"],
                "typical_features": ["Skill mastery", "Competition", "Real-world parallels"]
            },
            "Indie": {
                "subgenres": ["Art games", "Experimental", "Retro-inspired", "Minimalist"],
                "mechanics": ["Innovative mechanics", "Artistic expression", "Unique concepts"],
                "typical_features": ["Creative innovation", "Personal expression", "Niche appeal"]
            }
        }
    
    def _build_mechanics_taxonomy(self) -> Dict:
        """Build game mechanics taxonomy"""
        return {
            "Core Mechanics": {
                "Movement": ["Platforming", "Flying", "Swimming", "Parkour", "Teleportation"],
                "Combat": ["Melee", "Ranged", "Magic", "Stealth", "Turn-based", "Real-time"],
                "Progression": ["Leveling", "Skill trees", "Equipment upgrades", "Unlockables"],
                "Resource Management": ["Currency", "Inventory", "Energy/Stamina", "Time limits"]
            },
            "Social Mechanics": {
                "Multiplayer": ["Cooperative", "Competitive", "MMO", "Asynchronous"],
                "Communication": ["Voice chat", "Text chat", "Emotes", "Social features"]
            },
            "Meta Mechanics": {
                "Persistence": ["Save systems", "Cloud saves", "Cross-platform"],
                "Accessibility": ["Difficulty options", "Assist modes", "Customization"],
                "Monetization": ["Free-to-play", "DLC", "Microtransactions", "Season passes"]
            }
        }
    
    def _build_theme_ontology(self) -> Dict:
        """Build thematic ontology"""
        return {
            "Setting": {
                "Time Periods": ["Prehistoric", "Ancient", "Medieval", "Modern", "Future", "Post-apocalyptic"],
                "Locations": ["Fantasy worlds", "Space", "Urban", "Natural", "Abstract", "Historical"],
                "Atmosphere": ["Dark", "Light-hearted", "Serious", "Comedic", "Horror", "Romantic"]
            },
            "Narrative Themes": {
                "Coming of Age": ["Growth", "Discovery", "Responsibility"],
                "Conflict": ["War", "Survival", "Competition", "Cooperation"],
                "Exploration": ["Discovery", "Mystery", "Adventure", "Wonder"],
                "Relationships": ["Friendship", "Love", "Family", "Community"]
            }
        }
    
    def _build_audience_classification(self) -> Dict:
        """Build target audience classification"""
        return {
            "Age Groups": {
                "Children (E)": {"age_range": "0-12", "characteristics": ["Simple controls", "Educational", "Colorful"]},
                "Teens (T)": {"age_range": "13-17", "characteristics": ["Moderate complexity", "Social features", "Achievement-focused"]},
                "Adults (M)": {"age_range": "18+", "characteristics": ["Complex systems", "Mature themes", "Strategic depth"]}
            },
            "Experience Levels": {
                "Casual": {"time_commitment": "< 1 hour sessions", "complexity": "Low", "accessibility": "High"},
                "Enthusiast": {"time_commitment": "1-3 hour sessions", "complexity": "Medium", "accessibility": "Medium"},
                "Hardcore": {"time_commitment": "3+ hour sessions", "complexity": "High", "accessibility": "Low"}
            }
        }
    
    def _build_complexity_levels(self) -> Dict:
        """Build game complexity classification"""
        return {
            "Simple": {
                "learning_curve": "5-10 minutes",
                "systems": "1-2 core mechanics",
                "decision_complexity": "Low",
                "examples": ["Tetris", "Candy Crush", "Simple platformers"]
            },
            "Moderate": {
                "learning_curve": "30 minutes - 2 hours", 
                "systems": "3-5 interconnected mechanics",
                "decision_complexity": "Medium",
                "examples": ["Most RPGs", "Strategy games", "Action-adventures"]
            },
            "Complex": {
                "learning_curve": "Several hours to weeks",
                "systems": "5+ deeply interconnected systems",
                "decision_complexity": "High",
                "examples": ["Grand strategy", "Complex simulations", "Competitive esports"]
            }
        }
    
    def classify_game(self, game_data: Dict) -> GameTaxonomy:
        """Classify a game using the ontology system"""
        genres = self._extract_genres(game_data)
        mechanics = self._extract_mechanics(game_data)
        themes = self._extract_themes(game_data)
        
        primary_genre = self._determine_primary_genre(genres)
        complexity = self._determine_complexity(game_data, mechanics)
        audience = self._determine_audience(game_data, complexity)
        time_investment = self._determine_time_investment(game_data)
        
        return GameTaxonomy(
            primary_genre=primary_genre,
            sub_genres=genres,
            mechanics=mechanics,
            themes=themes,
            target_audience=audience,
            complexity_level=complexity,
            time_investment=time_investment,
            platform_preferences=game_data.get('platforms', [])
        )
    
    def _extract_genres(self, game_data: Dict) -> List[str]:
        """Extract and normalize genre information"""
        genres = []
        raw_genres = game_data.get('genres', [])
        
        if isinstance(raw_genres, str):
            raw_genres = [raw_genres]
        
        for genre in raw_genres:
            normalized = self._normalize_genre(genre)
            if normalized:
                genres.append(normalized)
        
        return list(set(genres))
    
    def _normalize_genre(self, genre: str) -> Optional[str]:
        """Normalize genre string to ontology standard"""
        if not genre or pd.isna(genre):
            return None
            
        genre = str(genre).strip().lower()
        
        # Genre mapping for normalization
        genre_mappings = {
            'action': 'Action',
            'adventure': 'Adventure', 
            'rpg': 'Role-Playing (RPG)',
            'role-playing': 'Role-Playing (RPG)',
            'strategy': 'Strategy',
            'simulation': 'Simulation',
            'puzzle': 'Puzzle',
            'sports': 'Sports & Racing',
            'racing': 'Sports & Racing',
            'indie': 'Indie',
            'casual': 'Indie'
        }
        
        for key, value in genre_mappings.items():
            if key in genre:
                return value
        
        return genre.title()
    
    def _extract_mechanics(self, game_data: Dict) -> List[str]:
        """Extract game mechanics from various data sources"""
        mechanics = []
        
        # From categories/tags
        categories = game_data.get('categories', [])
        tags = game_data.get('tags', [])
        
        combined_features = categories + tags if isinstance(categories, list) and isinstance(tags, list) else []
        
        for feature in combined_features:
            if isinstance(feature, str):
                mapped_mechanics = self._map_feature_to_mechanics(feature)
                mechanics.extend(mapped_mechanics)
        
        return list(set(mechanics))
    
    def _map_feature_to_mechanics(self, feature: str) -> List[str]:
        """Map game features to mechanics"""
        feature = feature.lower()
        mechanics = []
        
        mechanic_mappings = {
            'multiplayer': ['Multiplayer'],
            'co-op': ['Cooperative'],
            'pvp': ['Competitive'],
            'single-player': ['Single-player'],
            'story': ['Story progression'],
            'open world': ['Open world exploration'],
            'crafting': ['Crafting system'],
            'building': ['Construction mechanics'],
            'combat': ['Combat system'],
            'puzzle': ['Puzzle solving'],
            'platformer': ['Platforming'],
            'turn-based': ['Turn-based mechanics']
        }
        
        for key, mapped_mechanics in mechanic_mappings.items():
            if key in feature:
                mechanics.extend(mapped_mechanics)
        
        return mechanics
    
    def _extract_themes(self, game_data: Dict) -> List[str]:
        """Extract thematic elements"""
        themes = []
        
        # Extract from description, title, or tags
        description = str(game_data.get('description', ''))
        title = str(game_data.get('name', ''))
        
        theme_keywords = {
            'Fantasy': ['fantasy', 'magic', 'wizard', 'dragon', 'medieval'],
            'Sci-Fi': ['space', 'future', 'robot', 'alien', 'cyberpunk'],
            'Horror': ['horror', 'scary', 'zombie', 'monster', 'dark'],
            'Historical': ['historical', 'history', 'ancient', 'war'],
            'Modern': ['modern', 'contemporary', 'realistic'],
            'Post-Apocalyptic': ['post-apocalyptic', 'wasteland', 'survival']
        }
        
        combined_text = (description + ' ' + title).lower()
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in combined_text for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _determine_primary_genre(self, genres: List[str]) -> str:
        """Determine primary genre from list"""
        if not genres:
            return "Unclassified"
        
        # Priority order for primary genre
        genre_priority = [
            "Role-Playing (RPG)", "Strategy", "Action", "Adventure", 
            "Simulation", "Puzzle", "Sports & Racing", "Indie"
        ]
        
        for priority_genre in genre_priority:
            if priority_genre in genres:
                return priority_genre
        
        return genres[0]
    
    def _determine_complexity(self, game_data: Dict, mechanics: List[str]) -> str:
        """Determine game complexity level"""
        complexity_indicators = {
            'Simple': 0,
            'Moderate': 0, 
            'Complex': 0
        }
        
        # Mechanics count indicator
        if len(mechanics) <= 2:
            complexity_indicators['Simple'] += 2
        elif len(mechanics) <= 5:
            complexity_indicators['Moderate'] += 2
        else:
            complexity_indicators['Complex'] += 2
        
        # Genre complexity
        complex_genres = ['Strategy', 'Simulation', 'Role-Playing (RPG)']
        simple_genres = ['Puzzle', 'Indie']
        
        primary_genre = self._determine_primary_genre(game_data.get('genres', []))
        if primary_genre in complex_genres:
            complexity_indicators['Complex'] += 1
        elif primary_genre in simple_genres:
            complexity_indicators['Simple'] += 1
        else:
            complexity_indicators['Moderate'] += 1
        
        return max(complexity_indicators, key=complexity_indicators.get)
    
    def _determine_audience(self, game_data: Dict, complexity: str) -> str:
        """Determine target audience"""
        # Use age rating if available
        required_age = game_data.get('required_age', 0)
        
        if required_age >= 18:
            return "Adults (M)"
        elif required_age >= 13:
            return "Teens (T)"
        elif required_age > 0:
            return "Children (E)"
        
        # Fall back to complexity
        if complexity == 'Simple':
            return "Children (E)"
        elif complexity == 'Complex':
            return "Adults (M)"
        else:
            return "Teens (T)"
    
    def _determine_time_investment(self, game_data: Dict) -> str:
        """Determine typical time investment"""
        avg_playtime = game_data.get('average_playtime_forever', 0)
        
        if avg_playtime == 0:
            return "Unknown"
        elif avg_playtime < 60:  # Less than 1 hour average
            return "Short sessions (< 1 hour)"
        elif avg_playtime < 300:  # Less than 5 hours average
            return "Medium sessions (1-5 hours)"
        else:
            return "Long sessions (5+ hours)"
    
    def generate_recommendation_context(self, game_taxonomy: GameTaxonomy) -> Dict:
        """Generate rich context for recommendations"""
        return {
            "taxonomy": {
                "primary_genre": game_taxonomy.primary_genre,
                "sub_genres": game_taxonomy.sub_genres,
                "mechanics": game_taxonomy.mechanics,
                "themes": game_taxonomy.themes,
                "complexity": game_taxonomy.complexity_level,
                "audience": game_taxonomy.target_audience,
                "time_commitment": game_taxonomy.time_investment
            },
            "recommendation_criteria": {
                "genre_weight": 0.4,
                "mechanics_weight": 0.3,
                "theme_weight": 0.2,
                "audience_compatibility": 0.1
            },
            "hierarchy_depth": {
                "primary": game_taxonomy.primary_genre,
                "secondary": game_taxonomy.sub_genres[:2] if len(game_taxonomy.sub_genres) >= 2 else game_taxonomy.sub_genres,
                "mechanics_cluster": game_taxonomy.mechanics[:3] if len(game_taxonomy.mechanics) >= 3 else game_taxonomy.mechanics
            }
        }