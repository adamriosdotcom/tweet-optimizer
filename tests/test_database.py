"""
Tests for the database layer.
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta
from database import DatabaseError

def test_database_connection(test_db):
    """Test database connection"""
    assert test_db.conn is not None
    assert test_db.engine is not None

def test_insert_tweets(test_db, sample_tweets):
    """Test tweet insertion"""
    # Insert tweets
    rows_added = test_db.insert_tweets(sample_tweets)
    assert rows_added == len(sample_tweets)
    
    # Verify insertion
    with test_db.conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM tweets")
        count = cursor.fetchone()[0]
        assert count == len(sample_tweets)

def test_update_features(test_db, sample_tweets):
    """Test feature updates"""
    # First insert the tweets
    test_db.insert_tweets(sample_tweets)
    
    # Create feature updates
    feature_updates = pd.DataFrame({
        'uniqueid': sample_tweets['uniqueid'],
        'text_length': [10, 20, 30],
        'word_count': [2, 3, 4],
        'sentiment': [0.1, 0.2, 0.3]
    })
    
    # Update features
    test_db.update_features(feature_updates, ['text_length', 'word_count', 'sentiment'])
    
    # Verify updates
    with test_db.conn.cursor() as cursor:
        cursor.execute("""
            SELECT text_length, word_count, sentiment 
            FROM tweets 
            WHERE uniqueid = 'test1'
        """)
        result = cursor.fetchone()
        assert result[0] == 10
        assert result[1] == 2
        assert result[2] == 0.1

def test_get_tweet_by_id(test_db, sample_tweets):
    """Test retrieving tweet by ID"""
    # Insert tweets
    test_db.insert_tweets(sample_tweets)
    
    # Get tweet
    tweet = test_db.get_tweet_by_id('test1')
    assert tweet is not None
    assert tweet['text'] == 'Test tweet 1'
    assert tweet['author_username'] == 'user1'

def test_get_recent_tweets(test_db, sample_tweets):
    """Test retrieving recent tweets"""
    # Insert tweets
    test_db.insert_tweets(sample_tweets)
    
    # Get recent tweets
    recent_tweets = test_db.get_recent_tweets(limit=2)
    assert len(recent_tweets) == 2
    assert recent_tweets.iloc[0]['uniqueid'] == 'test1'  # Most recent

def test_database_error_handling(test_db):
    """Test database error handling"""
    # Test invalid query
    with pytest.raises(DatabaseError):
        test_db.get_recent_tweets(limit=-1)
    
    # Test invalid data insertion
    invalid_df = pd.DataFrame({
        'uniqueid': [None],  # Invalid primary key
        'text': ['Test']
    })
    with pytest.raises(DatabaseError):
        test_db.insert_tweets(invalid_df)

def test_batch_processing(test_db):
    """Test batch processing for large datasets"""
    # Create a larger dataset
    large_df = pd.DataFrame({
        'uniqueid': [f'test{i}' for i in range(150)],
        'text': [f'Test tweet {i}' for i in range(150)],
        'author_username': [f'user{i}' for i in range(150)],
        'created_at': [datetime.now() - timedelta(days=i) for i in range(150)],
        'like_count': [10] * 150,
        'retweet_count': [5] * 150,
        'reply_count': [2] * 150,
        'quote_count': [1] * 150,
        'view_count': [100] * 150,
        'bookmark_count': [3] * 150,
        'lang': ['en'] * 150,
        'topics': ['tech,ai'] * 150
    })
    
    # Insert with batch processing
    rows_added = test_db.insert_tweets(large_df)
    assert rows_added == len(large_df)
    
    # Verify all rows were inserted
    with test_db.conn.cursor() as cursor:
        cursor.execute("SELECT COUNT(*) FROM tweets")
        count = cursor.fetchone()[0]
        assert count == len(large_df) 