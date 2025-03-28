#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for the tweet analysis system.
"""

import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional

# Load environment variables
load_dotenv()

# MongoDB Configuration
MONGODB_URI = os.getenv('MONGODB_URI', 'mongodb://localhost:27017')
MONGODB_DB = os.getenv('MONGODB_DB', 'tweet_optimizer')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION', 'tweets')

# API Configuration
APIFY_API_KEY = os.getenv('APIFY_API_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Processing Configuration
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_TWEETS = 1000
DEFAULT_PROCESSING_INTERVAL = 3600  # 1 hour in seconds

# Model Configuration
MODEL_PATH = 'tweet_model.joblib'
VECTORIZER_PATH = 'tweet_vectorizer.joblib'

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Feature Configuration
ENGAGEMENT_WEIGHTS = {
    'retweet': 1,
    'favorite': 2,
    'reply': 3
}

# Clustering Configuration
N_CLUSTERS = 5
MAX_FEATURES = 1000

# Analysis Configuration
TRENDING_WINDOW_DAYS = 7
TOP_N_AUTHORS = 20
TOP_N_TOPICS = 20

# Dashboard Configuration
DASHBOARD_PORT = 8501
DASHBOARD_TITLE = "Tweet Optimizer Dashboard"
DASHBOARD_THEME = "light"

# File Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'
RESULTS_DIR = 'results'

# Create necessary directories
for directory in [DATA_DIR, MODELS_DIR, LOGS_DIR, RESULTS_DIR]:
    os.makedirs(directory, exist_ok=True)

class Settings(BaseSettings):
    """Application settings"""
    # MongoDB settings
    MONGODB_URI: str = MONGODB_URI
    MONGODB_DB: str = MONGODB_DB
    MONGODB_COLLECTION: str = MONGODB_COLLECTION
    
    # API settings
    APIFY_API_KEY: str = APIFY_API_KEY
    OPENAI_API_KEY: str = OPENAI_API_KEY
    API_KEY: str = os.getenv('API_KEY', 'your_api_key_here')
    API_HOST: str = os.getenv('API_HOST', '0.0.0.0')
    API_PORT: int = int(os.getenv('API_PORT', '8000'))
    
    # Database settings
    DB_HOST: str = os.getenv('DB_HOST', 'localhost')
    DB_PORT: int = int(os.getenv('DB_PORT', '5432'))
    DB_NAME: str = os.getenv('DB_NAME', 'twitter_data')
    DB_USER: str = os.getenv('DB_USER', 'adamrios')
    DB_PASSWORD: str = os.getenv('DB_PASSWORD', 'your_password_here')
    
    # Redis settings
    REDIS_HOST: str = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT: int = int(os.getenv('REDIS_PORT', '6379'))
    REDIS_DB: int = int(os.getenv('REDIS_DB', '0'))
    
    # Processing settings
    BATCH_SIZE: int = DEFAULT_BATCH_SIZE
    MAX_TWEETS: int = DEFAULT_MAX_TWEETS
    PROCESSING_INTERVAL: int = DEFAULT_PROCESSING_INTERVAL
    MAX_RETRIES: int = int(os.getenv('MAX_RETRIES', '3'))
    RETRY_DELAY: int = int(os.getenv('RETRY_DELAY', '2'))
    
    # Model settings
    MODEL_PATH: str = MODEL_PATH
    VECTORIZER_PATH: str = VECTORIZER_PATH
    
    # Feature settings
    ENGAGEMENT_WEIGHTS: dict = ENGAGEMENT_WEIGHTS
    
    # Clustering settings
    N_CLUSTERS: int = N_CLUSTERS
    MAX_FEATURES: int = MAX_FEATURES
    
    # Analysis settings
    TRENDING_WINDOW_DAYS: int = TRENDING_WINDOW_DAYS
    TOP_N_AUTHORS: int = TOP_N_AUTHORS
    TOP_N_TOPICS: int = TOP_N_TOPICS
    
    # Dashboard settings
    DASHBOARD_PORT: int = DASHBOARD_PORT
    DASHBOARD_TITLE: str = DASHBOARD_TITLE
    DASHBOARD_THEME: str = DASHBOARD_THEME
    
    # Twitter API settings
    TWITTER_API_KEY: str = os.getenv('TWITTER_API_KEY', 'your_twitter_api_key_here')
    TWITTER_API_SECRET: str = os.getenv('TWITTER_API_SECRET', 'your_twitter_api_secret_here')
    TWITTER_ACCESS_TOKEN: str = os.getenv('TWITTER_ACCESS_TOKEN', 'your_twitter_access_token_here')
    TWITTER_ACCESS_SECRET: str = os.getenv('TWITTER_ACCESS_SECRET', 'your_twitter_access_secret_here')
    
    # Optimization settings
    MAX_ITERATIONS: int = int(os.getenv('MAX_ITERATIONS', '5'))
    MAX_VARIATIONS: int = int(os.getenv('MAX_VARIATIONS', '10'))
    MIN_CONFIDENCE: float = float(os.getenv('MIN_CONFIDENCE', '0.5'))
    MAX_CONFIDENCE: float = float(os.getenv('MAX_CONFIDENCE', '0.95'))
    MAX_TOPICS: int = int(os.getenv('MAX_TOPICS', '100'))
    TOPIC_EXTRACTION_RETRIES: int = int(os.getenv('TOPIC_EXTRACTION_RETRIES', '3'))
    TOPIC_EXTRACTION_DELAY: int = int(os.getenv('TOPIC_EXTRACTION_DELAY', '2'))
    
    # Database path
    DB_PATH: str = os.getenv('DB_PATH', 'tweets.db')
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"  # Allow extra fields
    
    def get_database_uri(self) -> str:
        """Get the database URI."""
        return self.MONGODB_URI

# Global settings instance
settings = Settings() 