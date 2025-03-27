import sqlite3
import pandas as pd
from sqlalchemy import create_engine
import logging
from typing import List, Dict, Any, Optional
from config import settings
from init_db import init_database
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("database")

class DatabaseError(Exception):
    """Custom exception for database-related errors"""
    pass

class TwitterDatabase:
    def __init__(self, db_path='tweets.db'):
        """Initialize database connection using SQLite"""
        try:
            self.db_path = db_path
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.engine = create_engine(f'sqlite:///{db_path}')
            logger.info("Database connection established successfully")
        except sqlite3.Error as e:
            logger.error(f"Error connecting to database: {str(e)}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
        
    def __enter__(self):
        """Context manager entry"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")
        
    def setup_tables(self):
        """Create tables if they don't exist"""
        try:
            # Create tweets table if it doesn't exist
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS tweets (
                    uniqueid TEXT PRIMARY KEY,
                    text TEXT,
                    text_length INTEGER,
                    word_count INTEGER,
                    hashtag_count INTEGER,
                    mention_count INTEGER,
                    emoji_count INTEGER,
                    flesch_reading_ease REAL,
                    sentiment REAL,
                    subjectivity REAL,
                    topics TEXT,
                    engagement_score REAL,
                    popularity_index REAL,
                    account_age_days INTEGER,
                    follower_following_ratio REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Add processed column if it doesn't exist
            try:
                self.conn.execute("ALTER TABLE tweets ADD COLUMN processed TEXT")
                logger.info("Added processed column to tweets table")
            except sqlite3.OperationalError:
                logger.debug("processed column already exists")
            
            # Create indexes for better performance
            try:
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_processed ON tweets(processed)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at)")
                self.conn.execute("CREATE INDEX IF NOT EXISTS idx_tweets_engagement ON tweets(engagement_score)")
                logger.info("Created indexes for tweets table")
            except sqlite3.OperationalError as e:
                logger.warning(f"Some indexes already exist: {str(e)}")
            
            self.conn.commit()
            logger.info("Database tables and indexes created successfully")
        except Exception as e:
            logger.error(f"Failed to setup database tables: {str(e)}")
            raise DatabaseError(f"Database setup failed: {str(e)}")
    
    def insert_tweets(self, df: pd.DataFrame) -> int:
        """Insert tweets from dataframe with error handling"""
        try:
            # Clean and normalize column names
            df.columns = [col.lower().replace('.', '_') for col in df.columns]
            
            # Convert to proper types
            numeric_cols = ['like_count', 'retweet_count', 'reply_count', 'quote_count', 'view_count', 'bookmark_count']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Store raw JSON and extract core fields
            if 'raw_json' not in df.columns:
                df['raw_json'] = df.to_json(orient='records', lines=True).splitlines()
            
            # Insert using SQLAlchemy with chunking for large datasets
            chunk_size = settings.BATCH_SIZE
            total_rows = 0
            
            for i in range(0, len(df), chunk_size):
                chunk = df.iloc[i:i+chunk_size]
                chunk.to_sql('tweets', self.engine, if_exists='append', index=False, 
                           method='multi', chunksize=chunk_size)
                total_rows += len(chunk)
                logger.debug(f"Inserted {len(chunk)} rows (total: {total_rows})")
            
            return total_rows
            
        except Exception as e:
            logger.error(f"Failed to insert tweets: {str(e)}")
            raise DatabaseError(f"Tweet insertion failed: {str(e)}")
            
    def get_tweets_by_author(self, author_username: str, limit: int = 100) -> pd.DataFrame:
        """Get tweets by author username"""
        try:
            query = "SELECT * FROM tweets WHERE author_username = ? ORDER BY created_at DESC LIMIT ?"
            return pd.read_sql_query(query, self.conn, params=[author_username, limit])
        except Exception as e:
            logger.error(f"Failed to get tweets by author: {str(e)}")
            raise DatabaseError(f"Tweet retrieval failed: {str(e)}")
            
    def get_top_performing_tweets(self, metric: str = 'engagement_score', limit: int = 100) -> pd.DataFrame:
        """Get top performing tweets by specified metric"""
        try:
            query = f"SELECT * FROM tweets ORDER BY {metric} DESC LIMIT ?"
            return pd.read_sql_query(query, self.conn, params=[limit])
        except Exception as e:
            logger.error(f"Failed to get top performing tweets: {str(e)}")
            raise DatabaseError(f"Tweet retrieval failed: {str(e)}")
            
    def get_tweets_by_topic(self, topic: str, limit: int = 100) -> pd.DataFrame:
        """Get tweets by topic"""
        try:
            query = "SELECT * FROM tweets WHERE topics LIKE ? ORDER BY created_at DESC LIMIT ?"
            return pd.read_sql_query(query, self.conn, params=[f"%{topic}%", limit])
        except Exception as e:
            logger.error(f"Failed to get tweets by topic: {str(e)}")
            raise DatabaseError(f"Tweet retrieval failed: {str(e)}")
            
    def get_tweet_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the tweets in the database"""
        try:
            stats = {}
            
            # Total tweets
            stats['total_tweets'] = pd.read_sql_query("SELECT COUNT(*) as count FROM tweets", self.conn)['count'].iloc[0]
            
            # Average engagement
            stats['avg_engagement'] = pd.read_sql_query("SELECT AVG(engagement_score) as avg FROM tweets", self.conn)['avg'].iloc[0]
            
            # Top authors
            stats['top_authors'] = pd.read_sql_query("""
                SELECT author_username, COUNT(*) as tweet_count, AVG(engagement_score) as avg_engagement
                FROM tweets 
                GROUP BY author_username 
                ORDER BY avg_engagement DESC 
                LIMIT 10
            """, self.conn).to_dict('records')
            
            # Top topics
            stats['top_topics'] = pd.read_sql_query("""
                SELECT topics, COUNT(*) as count, AVG(engagement_score) as avg_engagement
                FROM tweets 
                WHERE topics IS NOT NULL AND topics != ''
                GROUP BY topics 
                ORDER BY avg_engagement DESC 
                LIMIT 10
            """, self.conn).to_dict('records')
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get tweet stats: {str(e)}")
            raise DatabaseError(f"Stats retrieval failed: {str(e)}")
    
    def update_features(self, df: pd.DataFrame, feature_columns: List[str], batch_size: int = None) -> None:
        """Update features for tweets with improved error handling"""
        if batch_size is None:
            batch_size = settings.BATCH_SIZE
            
        try:
            cursor = self.conn.cursor()
            logger.info(f"Starting feature update for {len(df)} rows with columns: {feature_columns}")
            
            # First ensure all columns exist
            for column in feature_columns:
                try:
                    cursor.execute(f"ALTER TABLE tweets ADD COLUMN {column} TEXT;")
                    logger.debug(f"Added column {column} to tweets table")
                except sqlite3.OperationalError:
                    logger.debug(f"Column {column} already exists")
                    pass
            
            # Process in batches
            total_updated = 0
            for i in range(0, len(df), batch_size):
                batch_df = df.iloc[i:i+batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}, size: {len(batch_df)}")
                
                # Update each row individually
                for _, row in batch_df.iterrows():
                    try:
                        set_clause = ', '.join([f'"{col}" = ?' for col in feature_columns])
                        values = [str(row[col]) for col in feature_columns]  # Convert all values to string
                        values.append(str(row['uniqueid']))  # Add uniqueid for WHERE clause
                        
                        update_query = f"""
                            UPDATE tweets 
                            SET {set_clause}
                            WHERE uniqueid = ?
                        """
                        
                        logger.info(f"Executing update for tweet {row['uniqueid']}")
                        logger.info(f"Query: {update_query}")
                        logger.info(f"Values: {values}")
                        
                        cursor.execute(update_query, values)
                        if cursor.rowcount > 0:
                            total_updated += 1
                            logger.info(f"Successfully updated tweet {row['uniqueid']}")
                        else:
                            logger.warning(f"No rows updated for tweet {row['uniqueid']}")
                            
                    except Exception as e:
                        logger.error(f"Error updating tweet {row['uniqueid']}: {str(e)}")
                        continue
                
                # Commit after each batch
                self.conn.commit()
                logger.info(f"Committed batch {i//batch_size + 1}, total updated so far: {total_updated}")
            
            logger.info(f"Feature update complete. Total rows updated: {total_updated}")
            return total_updated
                
        except Exception as e:
            logger.error(f"Failed to update features: {str(e)}")
            self.conn.rollback()  # Rollback on error
            raise DatabaseError(f"Feature update failed: {str(e)}")
    
    def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a tweet by its unique ID"""
        try:
            query = "SELECT * FROM tweets WHERE uniqueid = ?"
            return pd.read_sql_query(query, self.conn, params=(tweet_id,)).to_dict('records')[0]
        except Exception as e:
            logger.error(f"Failed to retrieve tweet {tweet_id}: {str(e)}")
            return None
    
    def get_recent_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent tweets with limit"""
        try:
            query = "SELECT * FROM tweets ORDER BY created_at DESC LIMIT ?"
            return pd.read_sql_query(query, self.conn, params=(limit,))
        except Exception as e:
            logger.error(f"Failed to retrieve recent tweets: {str(e)}")
            raise DatabaseError(f"Failed to retrieve recent tweets: {str(e)}")
    
    def get_tweet_features(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get features for a specific tweet"""
        try:
            query = """
            SELECT * FROM tweets WHERE uniqueid = ?
            """
            df = pd.read_sql_query(query, self.conn, params=(tweet_id,))
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return None
        except Exception as e:
            logger.error(f"Error getting tweet features: {str(e)}")
            raise DatabaseError(f"Failed to get tweet features: {str(e)}")
    
    def get_unprocessed_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Get tweets that need processing"""
        try:
            query = """
            SELECT * FROM tweets 
            WHERE text_length IS NULL OR sentiment IS NULL
            LIMIT ?
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            logger.info(f"Retrieved {len(df)} unprocessed tweets")
            return df
        except Exception as e:
            logger.error(f"Error getting unprocessed tweets: {str(e)}")
            raise DatabaseError(f"Failed to get unprocessed tweets: {str(e)}")
    
    def get_processed_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Get processed tweets for model training"""
        try:
            query = """
            SELECT * FROM tweets 
            WHERE text_length IS NOT NULL 
            AND sentiment IS NOT NULL 
            AND topic IS NOT NULL
            LIMIT ?
            """
            df = pd.read_sql_query(query, self.conn, params=(limit,))
            logger.info(f"Retrieved {len(df)} processed tweets")
            return df
        except Exception as e:
            logger.error(f"Error getting processed tweets: {str(e)}")
            raise DatabaseError(f"Failed to get processed tweets: {str(e)}")
    
    def get_similar_tweets(self, tweet_text: str, limit: int = 5) -> pd.DataFrame:
        """Get similar tweets from the database"""
        try:
            query = """
            SELECT * FROM tweets 
            WHERE text LIKE ? 
            AND text_length IS NOT NULL 
            AND sentiment IS NOT NULL 
            AND topic IS NOT NULL
            LIMIT ?
            """
            df = pd.read_sql_query(query, self.conn, params=(f"%{tweet_text[:50]}%", limit))
            logger.info(f"Retrieved {len(df)} similar tweets")
            return df
        except Exception as e:
            logger.error(f"Error getting similar tweets: {str(e)}")
            raise DatabaseError(f"Failed to get similar tweets: {str(e)}")
    
    def get_engagement_stats(self) -> Dict[str, float]:
        """Get engagement statistics for the dataset"""
        try:
            query = """
            SELECT 
                AVG(engagement_score) as avg_engagement,
                MAX(engagement_score) as max_engagement,
                MIN(engagement_score) as min_engagement,
                AVG(like_count) as avg_likes,
                AVG(retweet_count) as avg_retweets,
                AVG(reply_count) as avg_replies
            FROM tweets
            WHERE engagement_score IS NOT NULL
            """
            df = pd.read_sql_query(query, self.conn)
            if len(df) > 0:
                return df.iloc[0].to_dict()
            return {}
        except Exception as e:
            logger.error(f"Error getting engagement stats: {str(e)}")
            raise DatabaseError(f"Failed to get engagement stats: {str(e)}")
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of topics in the dataset"""
        try:
            query = """
            SELECT topics, COUNT(*) as count
            FROM tweets
            WHERE topics IS NOT NULL AND topics != ''
            GROUP BY topics
            """
            df = pd.read_sql_query(query, self.conn)
            
            # Parse topics from comma-separated string
            topics_dict = {}
            for _, row in df.iterrows():
                topics = [t.strip() for t in row['topics'].split(',') if t.strip()]
                for topic in topics:
                    topics_dict[topic] = topics_dict.get(topic, 0) + row['count']
            
            return dict(sorted(topics_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Error getting topic distribution: {str(e)}")
            raise DatabaseError(f"Failed to get topic distribution: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on correlation with engagement"""
        try:
            query = """
            SELECT 
                text_length, sentiment, hashtag_count, mention_count,
                url_count, has_media, emoji_count, word_count,
                avg_word_length, flesch_reading_ease, engagement_score
            FROM tweets
            WHERE text_length IS NOT NULL 
            AND sentiment IS NOT NULL
            """
            df = pd.read_sql_query(query, self.conn)
            
            if len(df) == 0:
                return {}
            
            # Calculate correlation with engagement score
            correlations = {}
            for column in df.columns:
                if column != 'engagement_score':
                    correlations[column] = abs(df[column].corr(df['engagement_score']))
            
            # Normalize correlations
            max_corr = max(correlations.values())
            if max_corr > 0:
                correlations = {k: v/max_corr for k, v in correlations.items()}
            
            return dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
            
        except Exception as e:
            logger.error(f"Error calculating feature importance: {str(e)}")
            raise DatabaseError(f"Failed to calculate feature importance: {str(e)}")
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get optimization history from file"""
        try:
            with open('optimization_history.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_optimization_history(self, history: List[Dict[str, Any]]) -> None:
        """Save optimization history to file"""
        try:
            with open('optimization_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            logger.info("Optimization history saved successfully")
        except Exception as e:
            logger.error(f"Error saving optimization history: {str(e)}")
            raise DatabaseError(f"Failed to save optimization history: {str(e)}") 