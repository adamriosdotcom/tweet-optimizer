#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration management for the tweet optimization system.
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_KEY: str = os.getenv("API_KEY", "default_key")
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Database Settings
    DB_PATH: str = "tweets.db"
    DB_HOST: str = os.getenv("DB_HOST", "localhost")
    DB_PORT: int = int(os.getenv("DB_PORT", "5432"))
    DB_NAME: str = os.getenv("DB_NAME", "twitter_data")
    DB_USER: str = os.getenv("DB_USER", "adamrios")
    DB_PASSWORD: str = os.getenv("DB_PASSWORD", "")
    
    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # Cache Settings
    CACHE_TTL: int = 3600  # 1 hour default TTL
    CACHE_PREDICT_TTL: int = 3600  # 1 hour for predictions
    CACHE_OPTIMIZE_TTL: int = 3600  # 1 hour for optimizations
    
    # OpenAI Settings
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    
    # Apify Settings
    APIFY_API_KEY: str
    
    # Feature engineering settings
    BATCH_SIZE: int = int(os.getenv('BATCH_SIZE', '100'))
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', '2'))
    
    # RL settings
    EXPLORATION_RATE: float = float(os.getenv('EXPLORATION_RATE', '0.1'))
    FEEDBACK_WINDOW_HOURS: int = int(os.getenv('FEEDBACK_WINDOW_HOURS', '24'))
    
    # Model settings
    MODEL_PATH: str = "xgboost_model.pkl"
    SCALER_PATH: str = "feature_scaler.pkl"
    FEATURE_NAMES_PATH: str = "feature_names.pkl"
    
    # Twitter settings
    TWITTER_API_KEY: str = os.getenv("TWITTER_API_KEY")
    TWITTER_API_SECRET: str = os.getenv("TWITTER_API_SECRET")
    TWITTER_ACCESS_TOKEN: str = os.getenv("TWITTER_ACCESS_TOKEN")
    TWITTER_ACCESS_SECRET: str = os.getenv("TWITTER_ACCESS_SECRET")
    
    # Feature engineering settings
    MAX_TOPICS: int = int(os.getenv('MAX_TOPICS', '100'))
    TOPIC_EXTRACTION_RETRIES: int = int(os.getenv('TOPIC_EXTRACTION_RETRIES', '3'))
    TOPIC_EXTRACTION_DELAY: int = int(os.getenv('TOPIC_EXTRACTION_DELAY', '2'))
    
    # Optimization settings
    MAX_ITERATIONS: int = int(os.getenv('MAX_ITERATIONS', '5'))
    MAX_VARIATIONS: int = int(os.getenv('MAX_VARIATIONS', '10'))
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.5'))
    MAX_CONFIDENCE: float = float(os.getenv('MAX_CONFIDENCE', '0.95'))
    
    @property
    def DATABASE_URL(self) -> str:
        """Get database URL"""
        return f"sqlite:///{self.DB_PATH}"
    
    class Config:
        """Pydantic config"""
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings() 