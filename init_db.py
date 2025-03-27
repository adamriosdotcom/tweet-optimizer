import sqlite3
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def init_database(db_path: str = 'tweets.db') -> None:
    """Initialize the SQLite database with the proper schema."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Drop existing table if it exists
        cursor.execute("DROP TABLE IF EXISTS tweets")
        
        # Create tweets table with proper schema
        cursor.execute("""
        CREATE TABLE tweets (
            uniqueid TEXT PRIMARY KEY,
            tweet_date TEXT,
            tweet_hour INTEGER,
            day_of_week TEXT,
            text TEXT,
            author_username TEXT,
            author_id TEXT,
            created_at TIMESTAMP,
            like_count INTEGER,
            retweet_count INTEGER,
            reply_count INTEGER,
            quote_count INTEGER,
            view_count INTEGER,
            bookmark_count INTEGER,
            lang TEXT,
            raw_json TEXT,
            
            -- Feature columns
            text_length INTEGER,
            word_count INTEGER,
            hashtag_count INTEGER,
            mention_count INTEGER,
            sentiment REAL,
            engagement_score REAL,
            likes_per_follower REAL,
            rolling_engagement REAL,
            follower_following_ratio REAL,
            account_age_days INTEGER,
            has_media INTEGER,
            retweet_like_ratio REAL,
            url_count INTEGER,
            distinct_hashtags INTEGER,
            unique_word_count INTEGER,
            avg_word_length REAL,
            punctuation_count INTEGER,
            uppercase_ratio REAL,
            emoji_count INTEGER,
            flesch_reading_ease REAL,
            subjectivity REAL,
            quote_influence REAL,
            response_time_minutes REAL,
            bio_length INTEGER,
            verified_engagement_avg REAL,
            has_location INTEGER,
            media_aspect_ratio REAL,
            birdwatch_flagged INTEGER,
            tweet_complexity INTEGER,
            daily_author_activity REAL,
            author_tweet_count TEXT,
            tweets_last_7_days INTEGER,
            tweets_last_30_days INTEGER,
            avg_tweets_per_day_7d REAL,
            avg_tweets_per_day_30d REAL,
            interaction_reciprocity INTEGER,
            followers_per_status REAL,
            follower_growth_rate REAL,
            popularity_index REAL,
            bio_url_count INTEGER,
            follower_growth_rate_7d REAL,
            follower_growth_rate_30d REAL,
            topics TEXT,
            topics_list TEXT,
            common_topic_count TEXT,
            jaccard_similarity TEXT
        )
        """)
        
        # Create indexes for commonly queried columns
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_author_username ON tweets(author_username)",
            "CREATE INDEX IF NOT EXISTS idx_created_at ON tweets(created_at)",
            "CREATE INDEX IF NOT EXISTS idx_like_count ON tweets(like_count)",
            "CREATE INDEX IF NOT EXISTS idx_topics ON tweets(topics)",
            "CREATE INDEX IF NOT EXISTS idx_tweet_date ON tweets(tweet_date)",
            "CREATE INDEX IF NOT EXISTS idx_engagement_score ON tweets(engagement_score)"
        ]
        
        for index_query in indexes:
            cursor.execute(index_query)
            
        conn.commit()
        logger.info("Database initialized successfully with proper schema")
        
    except Exception as e:
        logger.error(f"Failed to initialize database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_database() 