"""
GameMatch PyTorch & Hugging Face Integration
Advanced embeddings and semantic similarity for game recommendations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer, AutoModel, pipeline, 
    BertTokenizer, BertModel
)
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import pickle

logger = logging.getLogger(__name__)

class GameEmbeddingModel(nn.Module):
    """Custom PyTorch model for game embeddings"""
    
    def __init__(self, vocab_size=10000, embedding_dim=256, hidden_dim=512):
        super(GameEmbeddingModel, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.game_embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # Multi-head attention for genre relationships
        self.genre_attention = nn.MultiheadAttention(embedding_dim, num_heads=8)
        
        # Feed-forward networks
        self.genre_ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        self.mechanics_ff = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), 
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, embedding_dim)
        )
        
        # Classification heads
        self.genre_classifier = nn.Linear(embedding_dim, 20)  # Top 20 genres
        self.rating_predictor = nn.Linear(embedding_dim, 1)
        self.similarity_scorer = nn.Linear(embedding_dim * 2, 1)
        
    def forward(self, game_features, genre_features=None):
        # Game embeddings
        game_emb = self.game_embedding(game_features)
        
        # Apply attention if genre features available
        if genre_features is not None:
            attended, _ = self.genre_attention(game_emb, genre_features, genre_features)
            game_emb = game_emb + attended
        
        # Feature enhancement
        genre_enhanced = self.genre_ff(game_emb)
        mechanics_enhanced = self.mechanics_ff(game_emb)
        
        # Combine features
        combined = genre_enhanced + mechanics_enhanced
        
        return {
            'embeddings': combined,
            'genre_logits': self.genre_classifier(combined),
            'rating_pred': self.rating_predictor(combined)
        }

class HuggingFaceGameAnalyzer:
    """Hugging Face integration for game text analysis"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        
        # Pre-built pipelines
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        
        self.classification_pipeline = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli"
        )
        
        # Gaming-specific labels for classification
        self.genre_labels = [
            "action game", "adventure game", "role playing game", "strategy game",
            "simulation game", "puzzle game", "sports game", "racing game",
            "fighting game", "shooter game", "platformer game", "indie game"
        ]
        
        self.mechanic_labels = [
            "multiplayer", "single player", "cooperative", "competitive", 
            "open world", "story driven", "crafting", "building", "combat",
            "exploration", "puzzle solving", "character progression"
        ]
        
    def encode_game_description(self, descriptions: List[str]) -> np.ndarray:
        """Generate embeddings for game descriptions"""
        logger.info(f"Encoding {len(descriptions)} game descriptions...")
        
        # Tokenize and encode
        inputs = self.tokenizer(
            descriptions, 
            padding=True, 
            truncation=True, 
            return_tensors="pt",
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Use CLS token embeddings
            embeddings = outputs.last_hidden_state[:, 0, :].numpy()
        
        return embeddings
    
    def classify_game_genre(self, description: str) -> Dict:
        """Classify game genre using zero-shot classification"""
        try:
            result = self.classification_pipeline(description, self.genre_labels)
            return {
                "primary_genre": result['labels'][0],
                "confidence": result['scores'][0],
                "all_scores": dict(zip(result['labels'], result['scores']))
            }
        except Exception as e:
            logger.error(f"Genre classification failed: {e}")
            return {"primary_genre": "unknown", "confidence": 0.0}
    
    def classify_game_mechanics(self, description: str) -> Dict:
        """Classify game mechanics using zero-shot classification"""
        try:
            result = self.classification_pipeline(description, self.mechanic_labels)
            
            # Return top 3 mechanics with confidence > 0.3
            top_mechanics = []
            for label, score in zip(result['labels'][:5], result['scores'][:5]):
                if score > 0.3:
                    top_mechanics.append({"mechanic": label, "confidence": score})
            
            return {
                "mechanics": top_mechanics,
                "primary_mechanic": result['labels'][0] if result['scores'][0] > 0.3 else "general"
            }
        except Exception as e:
            logger.error(f"Mechanics classification failed: {e}")
            return {"mechanics": [], "primary_mechanic": "general"}
    
    def analyze_review_sentiment(self, reviews: List[str]) -> Dict:
        """Analyze sentiment of game reviews"""
        if not reviews or len(reviews) == 0:
            return {"overall_sentiment": "neutral", "confidence": 0.0}
        
        try:
            # Sample reviews if too many
            sample_reviews = reviews[:100] if len(reviews) > 100 else reviews
            
            sentiments = []
            for review in sample_reviews:
                if len(review.strip()) > 10:  # Skip very short reviews
                    result = self.sentiment_pipeline(review[:512])  # Truncate long reviews
                    sentiments.append(result[0])
            
            if not sentiments:
                return {"overall_sentiment": "neutral", "confidence": 0.0}
            
            # Aggregate sentiments
            positive_count = sum(1 for s in sentiments if s['label'] in ['POSITIVE', 'LABEL_2'])
            negative_count = sum(1 for s in sentiments if s['label'] in ['NEGATIVE', 'LABEL_0'])
            neutral_count = len(sentiments) - positive_count - negative_count
            
            total = len(sentiments)
            
            if positive_count / total > 0.6:
                overall = "positive"
                confidence = positive_count / total
            elif negative_count / total > 0.6:
                overall = "negative" 
                confidence = negative_count / total
            else:
                overall = "mixed"
                confidence = max(positive_count, negative_count, neutral_count) / total
            
            return {
                "overall_sentiment": overall,
                "confidence": confidence,
                "distribution": {
                    "positive": positive_count / total,
                    "negative": negative_count / total,
                    "neutral": neutral_count / total
                },
                "sample_size": total
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"overall_sentiment": "neutral", "confidence": 0.0}

# Usage example and testing functions
def test_pytorch_integration():
    """Test PyTorch model integration"""
    logger.info("Testing PyTorch GameEmbeddingModel...")
    
    # Create sample data
    vocab_size = 1000
    batch_size = 32
    seq_len = 10
    
    model = GameEmbeddingModel(vocab_size=vocab_size)
    
    # Sample input
    game_features = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        outputs = model(game_features)
    
    print(f"✅ PyTorch model test passed")
    print(f"   Embeddings shape: {outputs['embeddings'].shape}")
    print(f"   Genre logits shape: {outputs['genre_logits'].shape}")
    print(f"   Rating predictions shape: {outputs['rating_pred'].shape}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_pytorch_integration()