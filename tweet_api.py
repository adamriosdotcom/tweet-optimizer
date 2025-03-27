#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API for tweet prediction and optimization.
"""

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from typing import List, Optional
import logging
import uvicorn
from datetime import datetime, timedelta
from collections import defaultdict
from tweet_rl import TweetEnvironment, TweetOptimizer
from config import settings
from monitoring import (
    init_monitoring, RequestMetrics, record_prediction, record_optimization
)
from cache import cache

# Set up logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format=settings.LOG_FORMAT,
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tweet_api")

# Initialize environment and optimizer
env = TweetEnvironment()
optimizer = TweetOptimizer(env)

# Rate limiting
class RateLimiter:
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.requests = defaultdict(list)
    
    def is_rate_limited(self, client_id: str) -> bool:
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        # Clean old requests
        self.requests[client_id] = [req_time for req_time in self.requests[client_id] 
                                  if req_time > minute_ago]
        
        # Check if rate limited
        if len(self.requests[client_id]) >= self.requests_per_minute:
            return True
        
        # Add new request
        self.requests[client_id].append(now)
        return False

rate_limiter = RateLimiter()

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    if api_key != settings.API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

# Create FastAPI app
app = FastAPI(
    title="Tweet Optimization API",
    description="API for predicting and optimizing tweet engagement",
    version="1.0.0"
)

# Initialize monitoring
init_monitoring(app)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TweetRequest(BaseModel):
    """Request model for tweet prediction/optimization"""
    text: str = Field(..., min_length=1, max_length=280)
    iterations: Optional[int] = Field(default=1, ge=1, le=10)
    variations: Optional[int] = Field(default=5, ge=1, le=20)

class TweetPredictionResponse(BaseModel):
    """Response model for tweet prediction"""
    text: str
    predicted_engagement: float
    confidence: float

class TweetOptimizationResponse(BaseModel):
    """Response model for tweet optimization"""
    original_text: str
    original_engagement: float
    optimized_text: str
    optimized_engagement: float
    improvement: float
    history: List[dict]
    processing_time: float

@app.get("/")
def read_root():
    """Root endpoint"""
    return {
        "status": "ok",
        "message": "Tweet Optimization API",
        "version": "1.0.0"
    }

@app.post("/predict", response_model=TweetPredictionResponse)
async def predict_tweet(
    request: TweetRequest,
    api_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Predict engagement for a tweet"""
    # Rate limiting
    client_id = api_request.client.host
    if rate_limiter.is_rate_limited(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    # Check cache
    cache_key = f"predict:{request.text}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for prediction: {request.text[:50]}...")
        return cached_result
    
    with RequestMetrics("predict", "predict"):
        try:
            engagement = env.predict_engagement(request.text)
            
            # Calculate confidence based on feature values
            confidence = min(0.95, max(0.5, engagement / 100))  # Simple confidence calculation
            
            # Record metrics
            record_prediction(engagement)
            
            # Prepare response
            response = {
                "text": request.text,
                "predicted_engagement": float(engagement),
                "confidence": float(confidence)
            }
            
            # Cache the result
            cache.set(cache_key, response, settings.CACHE_PREDICT_TTL)
            
            return response
            
        except Exception as e:
            logger.error(f"Error predicting tweet engagement: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

@app.post("/optimize", response_model=TweetOptimizationResponse)
async def optimize_tweet(
    request: TweetRequest,
    api_request: Request,
    api_key: str = Depends(verify_api_key)
):
    """Optimize a tweet for engagement"""
    # Rate limiting
    client_id = api_request.client.host
    if rate_limiter.is_rate_limited(client_id):
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later."
        )
    
    # Check cache
    cache_key = f"optimize:{request.text}:{request.iterations}:{request.variations}"
    cached_result = cache.get(cache_key)
    if cached_result:
        logger.info(f"Cache hit for optimization: {request.text[:50]}...")
        return cached_result
    
    with RequestMetrics("optimize", "optimize"):
        try:
            # Get initial prediction
            initial_engagement = env.predict_engagement(request.text)
            
            # Optimize tweet
            optimized_tweet, history = optimizer.optimize_tweet(
                request.text, 
                n_iterations=request.iterations,
                n_variations=request.variations
            )
            
            # Get final prediction
            final_engagement = env.predict_engagement(optimized_tweet)
            
            # Calculate improvement percentage
            improvement = (final_engagement / initial_engagement - 1) * 100 if initial_engagement > 0 else 0
            
            # Record metrics
            record_optimization(improvement)
            
            # Prepare response
            response = {
                "original_text": request.text,
                "original_engagement": float(initial_engagement),
                "optimized_text": optimized_tweet,
                "optimized_engagement": float(final_engagement),
                "improvement": float(improvement),
                "history": history,
                "processing_time": 0.0  # Will be set by the RequestMetrics context manager
            }
            
            # Cache the result
            cache.set(cache_key, response, settings.CACHE_OPTIMIZE_TTL)
            
            return response
            
        except Exception as e:
            logger.error(f"Error optimizing tweet: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing request: {str(e)}"
            )

def main():
    """Run the API server"""
    uvicorn.run(
        "tweet_api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.LOG_LEVEL.lower()
    )

if __name__ == "__main__":
    main() 