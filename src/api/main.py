"""
GameMatch Production API
Enterprise-grade FastAPI microservice for game recommendations
"""

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
import logging
import time
from datetime import datetime
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Import GameMatch components
from models.advanced_rag_system import EnhancedRAGSystem
from models.gaming_ontology import GamingOntologySystem
from models.mlops_monitoring import RecommendationTracker, PerformanceMonitor
from data.dataset_loader import GameMatchDataLoader

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="GameMatch AI Recommendation Engine",
    description="Production API for personalized game recommendations using fine-tuned LLMs and RAG",
    version="2.1.0",
    contact={
        "name": "GameMatch API Team",
        "url": "https://github.com/gamematch/api",
        "email": "api@gamematch.ai",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global components (initialized on startup)
rag_system: EnhancedRAGSystem = None
ontology_system: GamingOntologySystem = None
tracker: RecommendationTracker = None
monitor: PerformanceMonitor = None
games_df: pd.DataFrame = None

# === API Models ===

class GameRequest(BaseModel):
    """Request model for game recommendations"""
    query: str = Field(..., description="Natural language query for game recommendations")
    user_id: Optional[str] = Field(None, description="User ID for personalization")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of recommendations")
    filters: Optional[Dict] = Field(None, description="Additional filters (genre, price, rating, etc.)")
    strategy: str = Field("hybrid_search", description="Retrieval strategy")
    include_reasoning: bool = Field(True, description="Include detailed reasoning in response")

class GameRecommendation(BaseModel):
    """Individual game recommendation"""
    game_id: int
    title: str
    genres: List[str]
    description: str
    rating: float
    price: float
    relevance_score: float
    reasoning: Optional[str] = None
    match_details: Optional[Dict] = None

class RecommendationResponse(BaseModel):
    """Complete recommendation response"""
    query: str
    user_id: Optional[str]
    recommendations: List[GameRecommendation]
    total_results: int
    processing_time_ms: float
    model_version: str
    strategy_used: str
    metadata: Dict

class UserFeedback(BaseModel):
    """User feedback on recommendations"""
    recommendation_id: str
    feedback_type: str = Field(..., pattern="^(positive|negative|neutral)$")
    rating: Optional[int] = Field(None, ge=1, le=5)
    clicked_games: Optional[List[int]] = None
    purchased_games: Optional[List[int]] = None
    time_spent_seconds: Optional[int] = None
    comments: Optional[str] = None

class HealthCheck(BaseModel):
    """System health status"""
    status: str
    version: str
    uptime_seconds: float
    components: Dict[str, bool]
    database_status: str
    model_status: str

# === Authentication ===

async def get_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Validate API key"""
    # In production, validate against database/service
    valid_keys = {"gamematch-api-key-2024", "demo-key-for-testing"}
    
    if credentials.credentials not in valid_keys:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return credentials.credentials

# === Startup/Shutdown Events ===

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    global rag_system, ontology_system, tracker, monitor, games_df
    
    logger.info("🚀 Starting GameMatch API...")
    
    try:
        # Load dataset
        logger.info("📦 Loading Steam games dataset...")
        data_loader = GameMatchDataLoader()
        games_df = data_loader.load_processed_data("data/processed/steam_games_processed.parquet")
        logger.info(f"✅ Loaded {len(games_df):,} games")
        
        # Initialize gaming ontology
        logger.info("🧠 Initializing gaming ontology...")
        ontology_system = GamingOntologySystem()
        logger.info("✅ Gaming ontology ready")
        
        # Initialize RAG system
        logger.info("🔍 Building RAG knowledge base...")
        rag_system = EnhancedRAGSystem(games_df.sample(n=min(5000, len(games_df))))
        logger.info("✅ RAG system ready")
        
        # Initialize monitoring
        logger.info("📊 Initializing MLOps monitoring...")
        tracker = RecommendationTracker()
        monitor = PerformanceMonitor(tracker)
        logger.info("✅ Monitoring systems ready")
        
        logger.info("🎉 GameMatch API fully initialized!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("👋 Shutting down GameMatch API...")

# === Health Check ===

@app.get("/health", response_model=HealthCheck, tags=["System"])
async def health_check():
    """System health check"""
    start_time = time.time()
    
    components = {
        "dataset_loaded": games_df is not None and len(games_df) > 0,
        "rag_system": rag_system is not None,
        "ontology_system": ontology_system is not None,
        "monitoring": tracker is not None and monitor is not None,
    }
    
    # Test database connection
    db_status = "connected"
    try:
        if tracker:
            # Simple connection test
            _ = tracker._get_connection()
    except Exception:
        db_status = "disconnected"
    
    processing_time = (time.time() - start_time) * 1000
    
    return HealthCheck(
        status="healthy" if all(components.values()) else "degraded",
        version="2.1.0",
        uptime_seconds=time.time() - startup_time if 'startup_time' in globals() else 0.0,
        components=components,
        database_status=db_status,
        model_status="ready" if rag_system else "loading"
    )

# === Main Recommendation Endpoint ===

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(
    request: GameRequest,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """
    Get personalized game recommendations using AI
    
    This endpoint uses:
    - Fine-tuned GPT-3.5-turbo model for understanding
    - Advanced RAG system for semantic search
    - Gaming ontology for hierarchical classification
    - Real-time MLOps monitoring
    """
    start_time = time.time()
    
    try:
        if not rag_system:
            raise HTTPException(status_code=503, detail="RAG system not initialized")
        
        # Query RAG system
        rag_context = rag_system.query(
            query=request.query,
            strategy=request.strategy,
            top_k=request.max_results,
            filters=request.filters,
            return_context=True
        )
        
        # Convert RAG results to API format
        recommendations = []
        for result in rag_context.retrieved_documents:
            doc = result.document
            
            recommendation = GameRecommendation(
                game_id=doc.game_id,
                title=doc.title,
                genres=doc.genres,
                description=doc.description[:300] + "..." if len(doc.description) > 300 else doc.description,
                rating=doc.rating,
                price=doc.price,
                relevance_score=result.relevance_score,
                reasoning=result.retrieval_reason if request.include_reasoning else None,
                match_details=result.query_match_details if request.include_reasoning else None
            )
            recommendations.append(recommendation)
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        response = RecommendationResponse(
            query=request.query,
            user_id=request.user_id,
            recommendations=recommendations,
            total_results=len(recommendations),
            processing_time_ms=processing_time_ms,
            model_version="GameMatch-v2.1-Production",
            strategy_used=request.strategy,
            metadata={
                "rag_context_quality": rag_context.context_quality_score,
                "search_terms": rag_context.search_metadata.get("search_terms", []),
                "filters_applied": request.filters or {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Log recommendation for MLOps monitoring (background task)
        background_tasks.add_task(
            log_recommendation_async,
            request.query,
            request.user_id or "anonymous",
            [r.game_id for r in recommendations],
            processing_time_ms,
            max([r.relevance_score for r in recommendations]) if recommendations else 0.0,
            "GameMatch-v2.1-Production"
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Recommendation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recommendation failed: {str(e)}")

# === Feedback Endpoint ===

@app.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    feedback: UserFeedback,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(get_api_key)
):
    """Submit user feedback on recommendations"""
    
    try:
        # Log feedback (background task)
        background_tasks.add_task(
            log_feedback_async,
            feedback.recommendation_id,
            feedback.feedback_type,
            feedback.rating,
            feedback.clicked_games,
            feedback.purchased_games,
            feedback.time_spent_seconds
        )
        
        return {"status": "success", "message": "Feedback recorded"}
        
    except Exception as e:
        logger.error(f"Feedback submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to record feedback")

# === Analytics Endpoints ===

@app.get("/analytics/games/search", tags=["Analytics"])
async def search_games(
    q: str = Query(..., description="Search query"),
    limit: int = Query(20, ge=1, le=100),
    api_key: str = Depends(get_api_key)
):
    """Search games by title or description"""
    
    if not games_df is not None:
        raise HTTPException(status_code=503, detail="Games dataset not loaded")
    
    # Simple text search
    mask = (
        games_df['Name'].str.contains(q, case=False, na=False) |
        games_df['Detailed description'].str.contains(q, case=False, na=False)
    )
    
    results = games_df[mask].head(limit)
    
    games = []
    for _, row in results.iterrows():
        games.append({
            "game_id": int(row.get('AppID', 0)),
            "title": row.get('Name', 'Unknown'),
            "genres": row.get('Genres', '').split(',') if row.get('Genres') else [],
            "price": float(row.get('Price', 0)),
            "rating": float(row.get('Review_Score', 0)),
            "description": row.get('Short description', '')[:200]
        })
    
    return {
        "query": q,
        "total_results": len(games),
        "games": games
    }

@app.get("/analytics/stats", tags=["Analytics"])
async def get_system_stats(api_key: str = Depends(get_api_key)):
    """Get system statistics"""
    
    stats = {
        "dataset": {
            "total_games": len(games_df) if games_df is not None else 0,
            "genres": len(set(games_df['Genres'].str.split(',').explode().dropna())) if games_df is not None else 0,
        },
        "rag_system": {
            "indexed_documents": len(rag_system.knowledge_index.documents) if rag_system else 0,
            "available_strategies": ["semantic_similarity", "categorical_filter", "hybrid_search", "collaborative_filter"]
        },
        "api": {
            "version": "2.1.0",
            "model_version": "GameMatch-v2.1-Production",
            "uptime": time.time() - startup_time if 'startup_time' in globals() else 0
        }
    }
    
    return stats

# === Background Tasks ===

async def log_recommendation_async(
    query: str, user_id: str, game_ids: List[int], 
    response_time_ms: float, confidence_score: float, model_version: str
):
    """Log recommendation asynchronously"""
    if tracker:
        try:
            tracker.log_recommendation(
                query=query,
                user_id=user_id,
                game_ids=game_ids,
                response_time_ms=response_time_ms,
                confidence_score=confidence_score,
                model_version=model_version
            )
        except Exception as e:
            logger.error(f"Failed to log recommendation: {e}")

async def log_feedback_async(
    recommendation_id: str, feedback_type: str, rating: Optional[int],
    clicked_games: Optional[List[int]], purchased_games: Optional[List[int]], 
    time_spent_seconds: Optional[int]
):
    """Log user feedback asynchronously"""
    if tracker:
        try:
            tracker.log_user_feedback(
                recommendation_id=int(recommendation_id),
                feedback_type=feedback_type,
                rating=rating,
                clicked_games=clicked_games,
                purchased_games=purchased_games,
                time_spent_seconds=time_spent_seconds
            )
        except Exception as e:
            logger.error(f"Failed to log feedback: {e}")

# Set startup time
startup_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)