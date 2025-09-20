"""
Advanced RAG (Retrieval-Augmented Generation) System for GameMatch
Comprehensive retrieval system with semantic search, vector embeddings, and context generation
"""

import pandas as pd
import numpy as np
import json
import pickle
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict
import re
import math
from datetime import datetime

# Core ML imports (with fallbacks)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.decomposition import TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class GameDocument:
    """Structured game document for RAG system"""
    game_id: int
    title: str
    description: str
    genres: List[str]
    mechanics: List[str]
    themes: List[str]
    price: float
    rating: float
    review_count: int
    tags: List[str]
    platforms: List[str]
    developer: str
    release_year: Optional[int]
    content_vector: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = None

@dataclass
class RetrievalResult:
    """Result from RAG retrieval"""
    document: GameDocument
    relevance_score: float
    retrieval_reason: str
    query_match_details: Dict[str, float]

@dataclass
class RAGContext:
    """Generated context for LLM"""
    query: str
    retrieved_games: List[RetrievalResult]
    context_summary: str
    structured_context: Dict[str, Any]
    retrieval_metadata: Dict[str, Any]

class GameKnowledgeIndex:
    """Advanced knowledge index for game data"""
    
    def __init__(self, embedding_dim: int = 512):
        self.embedding_dim = embedding_dim
        self.documents: List[GameDocument] = []
        self.tfidf_vectorizer = None
        self.svd_transformer = None  # Store SVD for query time
        self.content_vectors = None
        self.genre_index = defaultdict(list)
        self.price_index = defaultdict(list)
        self.rating_index = defaultdict(list)
        self.tag_index = defaultdict(list)
        self.developer_index = defaultdict(list)
        
        # Multi-modal embeddings
        self.text_embeddings = None
        self.numerical_features = None
        self.categorical_embeddings = None
        
        # Search indices
        self.inverted_index = defaultdict(set)
        self.phrase_index = defaultdict(set)
        
    def build_index(self, games_df: pd.DataFrame) -> None:
        """Build comprehensive knowledge index"""
        logger.info(f"Building RAG knowledge index for {len(games_df)} games...")
        
        # Convert DataFrame to GameDocuments
        self.documents = []
        for idx, row in games_df.iterrows():
            doc = self._create_game_document(idx, row)
            self.documents.append(doc)
            
            # Build categorical indices
            self._index_document(doc)
        
        # Build text embeddings
        if SKLEARN_AVAILABLE:
            self._build_text_embeddings()
            self._build_numerical_features()
            
        # Build search indices
        self._build_search_indices()
        
        logger.info(f"✅ Built RAG index with {len(self.documents)} documents")
        logger.info(f"   📝 Text embeddings: {self.text_embeddings.shape if self.text_embeddings is not None else 'None'}")
        logger.info(f"   🔢 Numerical features: {self.numerical_features.shape if self.numerical_features is not None else 'None'}")
        logger.info(f"   📋 Inverted index terms: {len(self.inverted_index)}")
    
    def _create_game_document(self, game_id: int, row: pd.Series) -> GameDocument:
        """Create structured document from game data"""
        
        # Extract text description
        description_parts = []
        if pd.notna(row.get('About the game')):
            description_parts.append(str(row['About the game']))
        if pd.notna(row.get('Name')):
            description_parts.append(f"Title: {row['Name']}")
        
        description = " | ".join(description_parts)
        
        # Extract structured data
        genres = row.get('Genres_List', []) if isinstance(row.get('Genres_List'), list) else []
        tags = row.get('Tags_List', []) if isinstance(row.get('Tags_List'), list) else []
        categories = row.get('Categories_List', []) if isinstance(row.get('Categories_List'), list) else []
        
        # Platforms
        platforms = []
        for platform in ['Windows', 'Mac', 'Linux']:
            if row.get(platform, False):
                platforms.append(platform)
        
        return GameDocument(
            game_id=game_id,
            title=str(row.get('Name', '')),
            description=description,
            genres=genres,
            mechanics=categories,  # Use categories as mechanics
            themes=[],  # Could be extracted from tags/description
            price=float(row.get('Price', 0)),
            rating=float(row.get('Review_Score', 0)),
            review_count=int(row.get('Total_Reviews', 0)),
            tags=tags,
            platforms=platforms,
            developer=str(row.get('Developers', '')),
            release_year=self._extract_year(row.get('Release date')),
            metadata={
                'commercial_viability': row.get('Commercial_Viability', 0),
                'wilson_score': row.get('Wilson_Score', 0),
                'content_richness': row.get('Content_Richness', 0)
            }
        )
    
    def _extract_year(self, date_str: Any) -> Optional[int]:
        """Extract year from date string"""
        if pd.isna(date_str):
            return None
        
        date_str = str(date_str)
        year_match = re.search(r'\b(19|20)\d{2}\b', date_str)
        return int(year_match.group()) if year_match else None
    
    def _index_document(self, doc: GameDocument) -> None:
        """Build categorical indices for document"""
        
        # Genre index
        for genre in doc.genres:
            self.genre_index[genre.lower()].append(doc.game_id)
        
        # Price index (bucketed)
        price_bucket = self._get_price_bucket(doc.price)
        self.price_index[price_bucket].append(doc.game_id)
        
        # Rating index (bucketed) 
        rating_bucket = self._get_rating_bucket(doc.rating)
        self.rating_index[rating_bucket].append(doc.game_id)
        
        # Tag index
        for tag in doc.tags:
            self.tag_index[tag.lower()].append(doc.game_id)
        
        # Developer index
        if doc.developer:
            self.developer_index[doc.developer.lower()].append(doc.game_id)
    
    def _get_price_bucket(self, price: float) -> str:
        """Get price bucket for indexing"""
        if price == 0:
            return "free"
        elif price < 5:
            return "budget"
        elif price < 15:
            return "economy"
        elif price < 30:
            return "standard"
        elif price < 60:
            return "premium"
        else:
            return "aaa"
    
    def _get_rating_bucket(self, rating: float) -> str:
        """Get rating bucket for indexing"""
        if rating >= 0.9:
            return "excellent"
        elif rating >= 0.8:
            return "very_good"
        elif rating >= 0.7:
            return "good"
        elif rating >= 0.6:
            return "fair"
        else:
            return "mixed"
    
    def _build_text_embeddings(self) -> None:
        """Build TF-IDF embeddings for text content"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Scikit-learn not available, skipping text embeddings")
            return
        
        # Combine all text content
        documents_text = []
        for doc in self.documents:
            text_parts = [doc.title, doc.description]
            text_parts.extend(doc.genres)
            text_parts.extend(doc.mechanics)
            text_parts.extend(doc.tags)
            
            combined_text = " ".join(str(part) for part in text_parts if part)
            documents_text.append(combined_text)
        
        # Build TF-IDF vectors with adaptive parameters
        num_docs = len(documents_text)
        min_df = min(2, max(1, num_docs // 10))  # Adaptive min_df
        max_df = 0.9 if num_docs < 10 else 0.7  # Higher max_df for small datasets
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=min(5000, num_docs * 100),  # Adaptive max_features
            stop_words='english',
            ngram_range=(1, 2),
            min_df=min_df,
            max_df=max_df
        )
        
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(documents_text)
        
        # Optional: Reduce dimensionality
        if tfidf_matrix.shape[1] > self.embedding_dim:
            self.svd_transformer = TruncatedSVD(n_components=self.embedding_dim)
            self.text_embeddings = self.svd_transformer.fit_transform(tfidf_matrix)
        else:
            self.text_embeddings = tfidf_matrix.toarray()
            self.svd_transformer = None
    
    def _build_numerical_features(self) -> None:
        """Build numerical feature matrix"""
        if not SKLEARN_AVAILABLE:
            return
        
        numerical_features = []
        for doc in self.documents:
            features = [
                doc.price,
                doc.rating,
                math.log1p(doc.review_count),  # Log-scaled review count
                len(doc.genres),
                len(doc.mechanics),
                len(doc.tags),
                len(doc.platforms),
                doc.release_year or 2000,  # Default year if missing
                doc.metadata.get('commercial_viability', 0),
                doc.metadata.get('wilson_score', 0),
                doc.metadata.get('content_richness', 0)
            ]
            numerical_features.append(features)
        
        self.numerical_features = np.array(numerical_features)
        
        # Normalize numerical features
        scaler = StandardScaler()
        self.numerical_features = scaler.fit_transform(self.numerical_features)
    
    def _build_search_indices(self) -> None:
        """Build inverted index for fast text search"""
        
        for doc in self.documents:
            # Tokenize all text content
            text_content = f"{doc.title} {doc.description}"
            text_content += " " + " ".join(doc.genres + doc.mechanics + doc.tags)
            
            # Simple tokenization
            tokens = re.findall(r'\b\w+\b', text_content.lower())
            
            for token in tokens:
                if len(token) > 2:  # Skip very short tokens
                    self.inverted_index[token].add(doc.game_id)
            
            # Build phrase index (2-grams)
            for i in range(len(tokens) - 1):
                phrase = f"{tokens[i]} {tokens[i+1]}"
                self.phrase_index[phrase].add(doc.game_id)

class AdvancedRAGRetriever:
    """Advanced retrieval system with multiple search strategies"""
    
    def __init__(self, knowledge_index: GameKnowledgeIndex):
        self.knowledge_index = knowledge_index
        self.retrieval_strategies = {
            'semantic_similarity': self._semantic_similarity_search,
            'categorical_filter': self._categorical_filter_search,
            'hybrid_search': self._hybrid_search,
            'collaborative_filter': self._collaborative_filter_search
        }
    
    def retrieve(
        self, 
        query: str, 
        strategy: str = 'hybrid_search',
        top_k: int = 10,
        filters: Dict[str, Any] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant games using specified strategy"""
        
        filters = filters or {}
        
        if strategy not in self.retrieval_strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        logger.info(f"Retrieving with strategy '{strategy}', top_k={top_k}")
        
        # Execute retrieval strategy
        results = self.retrieval_strategies[strategy](query, top_k, filters)
        
        # Post-process and rank results
        results = self._rerank_results(results, query)
        
        return results[:top_k]
    
    def _semantic_similarity_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Semantic similarity using embeddings"""
        
        if self.knowledge_index.text_embeddings is None:
            logger.warning("Text embeddings not available, falling back to keyword search")
            return self._keyword_search(query, top_k, filters)
        
        # Generate query embedding
        query_vector = self.knowledge_index.tfidf_vectorizer.transform([query])
        
        # Apply same transformation as training data
        if self.knowledge_index.svd_transformer is not None:
            query_vector = self.knowledge_index.svd_transformer.transform(query_vector)
        else:
            query_vector = query_vector.toarray()
        
        # Calculate similarities
        similarities = cosine_similarity(
            query_vector, 
            self.knowledge_index.text_embeddings
        )[0]
        
        # Get top results
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices:
            if len(results) >= top_k:
                break
                
            doc = self.knowledge_index.documents[idx]
            
            # Apply filters
            if not self._passes_filters(doc, filters):
                continue
                
            result = RetrievalResult(
                document=doc,
                relevance_score=float(similarities[idx]),
                retrieval_reason="Semantic similarity match",
                query_match_details={
                    "semantic_score": float(similarities[idx]),
                    "match_type": "embedding_similarity"
                }
            )
            results.append(result)
        
        return results
    
    def _categorical_filter_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Search using categorical indices"""
        
        candidate_ids = set()
        
        # Extract search terms from query
        query_lower = query.lower()
        
        # Search genre index
        for genre, game_ids in self.knowledge_index.genre_index.items():
            if genre in query_lower:
                candidate_ids.update(game_ids)
        
        # Search tag index
        for tag, game_ids in self.knowledge_index.tag_index.items():
            if tag in query_lower:
                candidate_ids.update(game_ids)
        
        # Apply filters
        if 'max_price' in filters:
            price_candidates = set()
            max_price = filters['max_price']
            for doc in self.knowledge_index.documents:
                if doc.price <= max_price:
                    price_candidates.add(doc.game_id)
            candidate_ids &= price_candidates
        
        if 'min_rating' in filters:
            rating_candidates = set()
            min_rating = filters['min_rating']
            for doc in self.knowledge_index.documents:
                if doc.rating >= min_rating:
                    rating_candidates.add(doc.game_id)
            candidate_ids &= rating_candidates
        
        # Convert to results
        results = []
        for game_id in list(candidate_ids)[:top_k]:
            doc = next(d for d in self.knowledge_index.documents if d.game_id == game_id)
            
            # Calculate relevance score based on matches
            relevance_score = self._calculate_categorical_relevance(doc, query)
            
            result = RetrievalResult(
                document=doc,
                relevance_score=relevance_score,
                retrieval_reason="Categorical index match",
                query_match_details={
                    "categorical_score": relevance_score,
                    "match_type": "category_filter"
                }
            )
            results.append(result)
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results
    
    def _hybrid_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Combine multiple search strategies"""
        
        # Get results from different strategies
        semantic_results = self._semantic_similarity_search(query, top_k, filters)
        categorical_results = self._categorical_filter_search(query, top_k, filters)
        
        # Combine and deduplicate
        combined_results = {}
        
        # Add semantic results with weight
        for result in semantic_results:
            game_id = result.document.game_id
            combined_results[game_id] = result
            combined_results[game_id].relevance_score *= 0.7  # Weight semantic
        
        # Add categorical results with weight
        for result in categorical_results:
            game_id = result.document.game_id
            if game_id in combined_results:
                # Combine scores
                combined_results[game_id].relevance_score += result.relevance_score * 0.3
                combined_results[game_id].retrieval_reason = "Hybrid: semantic + categorical"
            else:
                result.relevance_score *= 0.3  # Weight categorical
                combined_results[game_id] = result
        
        # Sort and return top results
        final_results = list(combined_results.values())
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return final_results[:top_k]
    
    def _collaborative_filter_search(
        self, 
        query: str, 
        top_k: int, 
        filters: Dict[str, Any]
    ) -> List[RetrievalResult]:
        """Collaborative filtering based on game relationships"""
        
        # Simple implementation - find games with similar numerical features
        if self.knowledge_index.numerical_features is None:
            return self._categorical_filter_search(query, top_k, filters)
        
        # Extract target game from query (if possible)
        target_games = []
        for doc in self.knowledge_index.documents:
            if doc.title.lower() in query.lower():
                target_games.append(doc)
        
        if not target_games:
            return self._hybrid_search(query, top_k, filters)
        
        # Use first found game as reference
        target_doc = target_games[0]
        target_idx = next(i for i, d in enumerate(self.knowledge_index.documents) if d.game_id == target_doc.game_id)
        
        target_features = self.knowledge_index.numerical_features[target_idx].reshape(1, -1)
        
        # Calculate feature similarities
        similarities = cosine_similarity(
            target_features,
            self.knowledge_index.numerical_features
        )[0]
        
        # Get top similar games
        top_indices = np.argsort(similarities)[::-1]
        
        results = []
        for idx in top_indices[1:top_k+1]:  # Skip self
            doc = self.knowledge_index.documents[idx]
            
            if not self._passes_filters(doc, filters):
                continue
            
            result = RetrievalResult(
                document=doc,
                relevance_score=float(similarities[idx]),
                retrieval_reason=f"Similar to {target_doc.title}",
                query_match_details={
                    "collaborative_score": float(similarities[idx]),
                    "reference_game": target_doc.title,
                    "match_type": "collaborative_filter"
                }
            )
            results.append(result)
        
        return results
    
    def _keyword_search(self, query: str, top_k: int, filters: Dict[str, Any]) -> List[RetrievalResult]:
        """Fallback keyword search using inverted index"""
        
        query_tokens = re.findall(r'\b\w+\b', query.lower())
        candidate_scores = defaultdict(float)
        
        # Score based on term frequency
        for token in query_tokens:
            if token in self.knowledge_index.inverted_index:
                for game_id in self.knowledge_index.inverted_index[token]:
                    candidate_scores[game_id] += 1.0
        
        # Score based on phrase matches
        for i in range(len(query_tokens) - 1):
            phrase = f"{query_tokens[i]} {query_tokens[i+1]}"
            if phrase in self.knowledge_index.phrase_index:
                for game_id in self.knowledge_index.phrase_index[phrase]:
                    candidate_scores[game_id] += 2.0  # Higher weight for phrases
        
        # Convert to results
        results = []
        for game_id, score in sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]:
            doc = next(d for d in self.knowledge_index.documents if d.game_id == game_id)
            
            if not self._passes_filters(doc, filters):
                continue
            
            # Normalize score
            normalized_score = min(1.0, score / len(query_tokens))
            
            result = RetrievalResult(
                document=doc,
                relevance_score=normalized_score,
                retrieval_reason="Keyword match",
                query_match_details={
                    "keyword_score": normalized_score,
                    "match_type": "keyword_search"
                }
            )
            results.append(result)
        
        return results
    
    def _passes_filters(self, doc: GameDocument, filters: Dict[str, Any]) -> bool:
        """Check if document passes all filters"""
        
        if 'max_price' in filters and doc.price > filters['max_price']:
            return False
        
        if 'min_rating' in filters and doc.rating < filters['min_rating']:
            return False
        
        if 'genres' in filters:
            required_genres = filters['genres']
            if not any(genre.lower() in [g.lower() for g in doc.genres] for genre in required_genres):
                return False
        
        if 'platforms' in filters:
            required_platforms = filters['platforms']
            if not any(platform in doc.platforms for platform in required_platforms):
                return False
        
        return True
    
    def _calculate_categorical_relevance(self, doc: GameDocument, query: str) -> float:
        """Calculate relevance score for categorical matches"""
        query_lower = query.lower()
        score = 0.0
        
        # Genre matches
        for genre in doc.genres:
            if genre.lower() in query_lower:
                score += 0.3
        
        # Tag matches
        for tag in doc.tags:
            if tag.lower() in query_lower:
                score += 0.1
        
        # Title match
        if any(word in doc.title.lower() for word in query_lower.split()):
            score += 0.5
        
        return min(1.0, score)
    
    def _rerank_results(self, results: List[RetrievalResult], query: str) -> List[RetrievalResult]:
        """Post-process and rerank results"""
        
        # Apply quality boosting
        for result in results:
            doc = result.document
            
            # Boost high-quality games
            quality_boost = 0.0
            if doc.rating > 0.8:
                quality_boost += 0.1
            if doc.review_count > 1000:
                quality_boost += 0.05
            if doc.metadata.get('commercial_viability', 0) > 0.7:
                quality_boost += 0.05
            
            result.relevance_score += quality_boost
        
        # Sort by final score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results

class RAGContextGenerator:
    """Generate structured context for LLM from retrieved documents"""
    
    def __init__(self, max_context_games: int = 10, max_context_length: int = 2000):
        self.max_context_games = max_context_games
        self.max_context_length = max_context_length
    
    def generate_context(
        self, 
        query: str, 
        retrieved_results: List[RetrievalResult]
    ) -> RAGContext:
        """Generate comprehensive context for LLM"""
        
        # Limit results
        top_results = retrieved_results[:self.max_context_games]
        
        # Generate context summary
        context_summary = self._generate_context_summary(query, top_results)
        
        # Generate structured context
        structured_context = self._generate_structured_context(top_results)
        
        # Generate retrieval metadata
        retrieval_metadata = self._generate_retrieval_metadata(top_results)
        
        return RAGContext(
            query=query,
            retrieved_games=top_results,
            context_summary=context_summary,
            structured_context=structured_context,
            retrieval_metadata=retrieval_metadata
        )
    
    def _generate_context_summary(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate natural language context summary"""
        
        if not results:
            return f"No relevant games found for query: {query}"
        
        # Extract key themes
        all_genres = []
        price_ranges = []
        ratings = []
        
        for result in results:
            doc = result.document
            all_genres.extend(doc.genres)
            price_ranges.append(doc.price)
            ratings.append(doc.rating)
        
        # Analyze themes
        from collections import Counter
        genre_counts = Counter(all_genres)
        top_genres = [genre for genre, _ in genre_counts.most_common(3)]
        
        avg_price = np.mean(price_ranges)
        avg_rating = np.mean(ratings)
        
        summary = f"Found {len(results)} relevant games for '{query}'. "
        
        if top_genres:
            summary += f"Primary genres: {', '.join(top_genres)}. "
        
        summary += f"Average price: ${avg_price:.2f}, Average rating: {avg_rating:.1%}."
        
        # Add top recommendations
        if len(results) >= 3:
            top_3 = results[:3]
            summary += f" Top recommendations: {', '.join([r.document.title for r in top_3])}."
        
        return summary
    
    def _generate_structured_context(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate structured context data"""
        
        games_data = []
        for result in results:
            doc = result.document
            
            game_context = {
                "title": doc.title,
                "genres": doc.genres,
                "price": doc.price,
                "rating": doc.rating,
                "review_count": doc.review_count,
                "platforms": doc.platforms,
                "relevance_score": result.relevance_score,
                "match_reason": result.retrieval_reason,
                "key_features": doc.mechanics[:3],  # Top 3 features
                "tags": doc.tags[:5]  # Top 5 tags
            }
            games_data.append(game_context)
        
        # Aggregate statistics
        aggregated_stats = {
            "total_games": len(results),
            "avg_price": float(np.mean([r.document.price for r in results])),
            "avg_rating": float(np.mean([r.document.rating for r in results])),
            "price_range": {
                "min": float(min(r.document.price for r in results)),
                "max": float(max(r.document.price for r in results))
            },
            "genre_distribution": self._get_genre_distribution(results),
            "platform_coverage": self._get_platform_coverage(results)
        }
        
        return {
            "games": games_data,
            "statistics": aggregated_stats,
            "retrieval_quality": {
                "avg_relevance": float(np.mean([r.relevance_score for r in results])),
                "min_relevance": float(min(r.relevance_score for r in results)),
                "relevance_variance": float(np.var([r.relevance_score for r in results]))
            }
        }
    
    def _get_genre_distribution(self, results: List[RetrievalResult]) -> Dict[str, int]:
        """Get genre distribution in results"""
        from collections import Counter
        
        all_genres = []
        for result in results:
            all_genres.extend(result.document.genres)
        
        return dict(Counter(all_genres).most_common(10))
    
    def _get_platform_coverage(self, results: List[RetrievalResult]) -> Dict[str, int]:
        """Get platform coverage in results"""
        from collections import Counter
        
        all_platforms = []
        for result in results:
            all_platforms.extend(result.document.platforms)
        
        return dict(Counter(all_platforms))
    
    def _generate_retrieval_metadata(self, results: List[RetrievalResult]) -> Dict[str, Any]:
        """Generate metadata about retrieval process"""
        
        retrieval_strategies = set()
        match_types = set()
        
        for result in results:
            retrieval_strategies.add(result.retrieval_reason)
            match_types.add(result.query_match_details.get('match_type', 'unknown'))
        
        return {
            "retrieval_timestamp": datetime.now().isoformat(),
            "strategies_used": list(retrieval_strategies),
            "match_types": list(match_types),
            "result_count": len(results),
            "quality_indicators": {
                "high_relevance_count": sum(1 for r in results if r.relevance_score > 0.7),
                "medium_relevance_count": sum(1 for r in results if 0.4 <= r.relevance_score <= 0.7),
                "low_relevance_count": sum(1 for r in results if r.relevance_score < 0.4)
            }
        }

class EnhancedRAGSystem:
    """Complete RAG system orchestrator"""
    
    def __init__(self, games_df: pd.DataFrame = None):
        self.knowledge_index = GameKnowledgeIndex()
        self.retriever = None
        self.context_generator = RAGContextGenerator()
        
        if games_df is not None:
            self.build_knowledge_base(games_df)
    
    def build_knowledge_base(self, games_df: pd.DataFrame) -> None:
        """Build the complete knowledge base"""
        logger.info("Building enhanced RAG system...")
        
        self.knowledge_index.build_index(games_df)
        self.retriever = AdvancedRAGRetriever(self.knowledge_index)
        
        logger.info("✅ Enhanced RAG system ready")
    
    def query(
        self, 
        query: str, 
        strategy: str = 'hybrid_search',
        top_k: int = 10,
        filters: Dict[str, Any] = None,
        return_context: bool = True
    ) -> Union[List[RetrievalResult], RAGContext]:
        """Main query interface"""
        
        if self.retriever is None:
            raise ValueError("Knowledge base not built. Call build_knowledge_base() first.")
        
        # Retrieve relevant documents
        results = self.retriever.retrieve(query, strategy, top_k, filters)
        
        if not return_context:
            return results
        
        # Generate context for LLM
        context = self.context_generator.generate_context(query, results)
        
        return context
    
    def save_index(self, index_name: str = "default_rag_index") -> None:
        """Save the built index to MSSQL database"""
        # Note: For production systems, we would save embeddings to MSSQL using varbinary(max)
        # For this demo, we'll keep indices in memory and log the capability
        logger.info(f"RAG index '{index_name}' ready for MSSQL persistence")
        logger.info(f"Index contains {len(self.knowledge_index.documents)} documents with embeddings")
    
    def load_index(self, index_name: str = "default_rag_index") -> None:
        """Load index from MSSQL database"""
        # Note: For production systems, we would load embeddings from MSSQL
        # For this demo, we'll rebuild from the parquet data
        logger.info(f"RAG index '{index_name}' loading capability ready for MSSQL")
        logger.info("In production, this would load embeddings from MSSQL varbinary columns")

# Testing and demonstration functions
def test_rag_system():
    """Test the complete RAG system"""
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing Enhanced RAG System...")
    
    # Create sample data
    sample_games = pd.DataFrame([
        {
            'Name': 'The Witcher 3: Wild Hunt',
            'Genres_List': ['RPG', 'Adventure'],
            'About the game': 'Epic fantasy RPG with deep storytelling',
            'Price': 39.99,
            'Review_Score': 0.96,
            'Total_Reviews': 487235,
            'Windows': True,
            'Tags_List': ['Open World', 'Fantasy', 'Story Rich']
        },
        {
            'Name': 'Stardew Valley',
            'Genres_List': ['Simulation', 'Indie'],
            'About the game': 'Relaxing farming simulation game',
            'Price': 14.99,
            'Review_Score': 0.98,
            'Total_Reviews': 423847,
            'Windows': True,
            'Tags_List': ['Farming', 'Relaxing', 'Multiplayer']
        }
    ])
    
    # Initialize RAG system
    rag_system = EnhancedRAGSystem(sample_games)
    
    # Test queries
    test_queries = [
        "fantasy RPG games",
        "relaxing farming simulation",
        "games similar to Witcher 3",
        "cheap indie games under $20"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        try:
            context = rag_system.query(query, strategy='hybrid_search', top_k=3)
            print(f"   Found {len(context.retrieved_games)} relevant games")
            print(f"   Context summary: {context.context_summary}")
            
            for i, result in enumerate(context.retrieved_games):
                print(f"   {i+1}. {result.document.title} (relevance: {result.relevance_score:.3f})")
        
        except Exception as e:
            print(f"   Error: {e}")
    
    print("\n✅ RAG system test completed!")

if __name__ == "__main__":
    test_rag_system()