#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration settings for the tweet analysis system.
"""

import os
from dotenv import load_dotenv
from pydantic import BaseSettings
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
    
    # Processing settings
    BATCH_SIZE: int = DEFAULT_BATCH_SIZE
    MAX_TWEETS: int = DEFAULT_MAX_TWEETS
    PROCESSING_INTERVAL: int = DEFAULT_PROCESSING_INTERVAL
    
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
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    def get_database_uri(self) -> str:
        """Get the database URI."""
        return self.MONGODB_URI

# Global settings instance
settings = Settings() 