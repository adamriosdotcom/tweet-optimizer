"""
Pytest configuration and fixtures.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from database import TwitterDatabase
from tweet_rl import TweetEnvironment, TweetOptimizer
from config import settings

@pytest.fixture
def test_db():
    """Create a test database connection"""
    # Use a test database name
    settings.DB_NAME = 'test_twitter_data'
    db = TwitterDatabase()
    db.setup_tables()
    yield db
    # Cleanup after tests
    with db.conn.cursor() as cursor:
        cursor.execute("DROP TABLE IF EXISTS tweets CASCADE;")
        cursor.execute("DROP TABLE IF EXISTS temp_features CASCADE;")
    db.conn.commit()
    db.conn.close()

@pytest.fixture
def sample_tweets():
    """Create sample tweet data for testing"""
    return pd.DataFrame({
        'uniqueid': ['test1', 'test2', 'test3'],
        'text': ['Test tweet 1', 'Test tweet 2', 'Test tweet 3'],
        'author_username': ['user1', 'user2', 'user3'],
        'created_at': [datetime.now() - timedelta(days=i) for i in range(3)],
        'like_count': [10, 20, 30],
        'retweet_count': [5, 10, 15],
        'reply_count': [2, 4, 6],
        'quote_count': [1, 2, 3],
        'view_count': [100, 200, 300],
        'bookmark_count': [3, 6, 9],
        'lang': ['en', 'en', 'en'],
        'topics': ['tech,ai', 'data,science', 'ml,dl']
    })

@pytest.fixture
def tweet_env():
    """Create a test tweet environment"""
    return TweetEnvironment()

@pytest.fixture
def tweet_optimizer(tweet_env):
    """Create a test tweet optimizer"""
    return TweetOptimizer(tweet_env) 