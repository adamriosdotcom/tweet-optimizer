from pymongo import MongoClient
from pymongo.errors import PyMongoError
import pandas as pd
import logging
from typing import List, Dict, Any, Optional
from config import settings
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
    def __init__(self):
        """Initialize MongoDB connection"""
        try:
            self.client = MongoClient(settings.MONGODB_URI)
            self.db = self.client[settings.MONGODB_DB]
            self.tweets = self.db.tweets
            
            # Create indexes for better performance
            self.tweets.create_index("uniqueid", unique=True)
            self.tweets.create_index("processed")
            self.tweets.create_index("created_at")
            self.tweets.create_index("engagement_score")
            
            logger.info("MongoDB connection established successfully")
        except PyMongoError as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise DatabaseError(f"Failed to connect to database: {str(e)}")
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if hasattr(self, 'client'):
            self.client.close()
            logger.info("MongoDB connection closed")
    
    def insert_tweets(self, df: pd.DataFrame) -> int:
        """Insert tweets from dataframe with error handling"""
        try:
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Insert using bulk operations for better performance
            result = self.tweets.insert_many(records, ordered=False)
            return len(result.inserted_ids)
            
        except Exception as e:
            logger.error(f"Failed to insert tweets: {str(e)}")
            raise DatabaseError(f"Tweet insertion failed: {str(e)}")
    
    def get_tweet_stats(self) -> Dict[str, Any]:
        """Get basic statistics about the tweets in the database"""
        try:
            stats = {}
            
            # Total tweets
            stats['total_tweets'] = self.tweets.count_documents({})
            
            # Average engagement
            pipeline = [
                {"$group": {"_id": None, "avg": {"$avg": "$engagement_score"}}}
            ]
            result = list(self.tweets.aggregate(pipeline))
            stats['avg_engagement'] = result[0]['avg'] if result else 0
            
            # Top authors
            pipeline = [
                {"$group": {
                    "_id": "$author_username",
                    "tweet_count": {"$sum": 1},
                    "avg_engagement": {"$avg": "$engagement_score"}
                }},
                {"$sort": {"avg_engagement": -1}},
                {"$limit": 10}
            ]
            stats['top_authors'] = list(self.tweets.aggregate(pipeline))
            
            # Top topics
            pipeline = [
                {"$group": {
                    "_id": "$topics",
                    "count": {"$sum": 1},
                    "avg_engagement": {"$avg": "$engagement_score"}
                }},
                {"$sort": {"avg_engagement": -1}},
                {"$limit": 10}
            ]
            stats['top_topics'] = list(self.tweets.aggregate(pipeline))
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get tweet stats: {str(e)}")
            raise DatabaseError(f"Stats retrieval failed: {str(e)}")
    
    def update_features(self, df: pd.DataFrame, feature_columns: List[str]) -> int:
        """Update features for tweets with improved error handling"""
        try:
            total_updated = 0
            
            # Convert DataFrame to list of dictionaries
            records = df.to_dict('records')
            
            # Update each record
            for record in records:
                try:
                    # Prepare update document
                    update_doc = {k: record[k] for k in feature_columns}
                    
                    # Update the document
                    result = self.tweets.update_one(
                        {"uniqueid": record['uniqueid']},
                        {"$set": update_doc}
                    )
                    
                    if result.modified_count > 0:
                        total_updated += 1
                        logger.info(f"Successfully updated tweet {record['uniqueid']}")
                    else:
                        logger.warning(f"No rows updated for tweet {record['uniqueid']}")
                        
                except Exception as e:
                    logger.error(f"Error updating tweet {record['uniqueid']}: {str(e)}")
                    continue
            
            logger.info(f"Feature update complete. Total rows updated: {total_updated}")
            return total_updated
                
        except Exception as e:
            logger.error(f"Failed to update features: {str(e)}")
            raise DatabaseError(f"Feature update failed: {str(e)}")
    
    def get_tweet_by_id(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a tweet by its unique ID"""
        try:
            return self.tweets.find_one({"uniqueid": tweet_id})
        except Exception as e:
            logger.error(f"Failed to retrieve tweet {tweet_id}: {str(e)}")
            return None
    
    def get_recent_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Retrieve recent tweets with limit"""
        try:
            cursor = self.tweets.find().sort("created_at", -1).limit(limit)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Failed to retrieve recent tweets: {str(e)}")
            raise DatabaseError(f"Failed to retrieve recent tweets: {str(e)}")
    
    def get_tweet_features(self, tweet_id: str) -> Optional[Dict[str, Any]]:
        """Get features for a specific tweet"""
        try:
            return self.tweets.find_one({"uniqueid": tweet_id})
        except Exception as e:
            logger.error(f"Error getting tweet features: {str(e)}")
            raise DatabaseError(f"Failed to get tweet features: {str(e)}")
    
    def get_unprocessed_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Get tweets that need processing"""
        try:
            query = {
                "$or": [
                    {"text_length": None},
                    {"sentiment": None},
                    {"processed": None},
                    {"processed": ""}
                ]
            }
            cursor = self.tweets.find(query).limit(limit)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Error getting unprocessed tweets: {str(e)}")
            raise DatabaseError(f"Failed to get unprocessed tweets: {str(e)}")
    
    def get_processed_tweets(self, limit: int = 100) -> pd.DataFrame:
        """Get processed tweets for model training"""
        try:
            query = {
                "text_length": {"$ne": None},
                "sentiment": {"$ne": None},
                "topics": {"$ne": None}
            }
            cursor = self.tweets.find(query).limit(limit)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Error getting processed tweets: {str(e)}")
            raise DatabaseError(f"Failed to get processed tweets: {str(e)}")
    
    def get_similar_tweets(self, tweet_text: str, limit: int = 5) -> pd.DataFrame:
        """Get similar tweets from the database"""
        try:
            query = {
                "text": {"$regex": tweet_text[:50], "$options": "i"},
                "text_length": {"$ne": None},
                "sentiment": {"$ne": None},
                "topics": {"$ne": None}
            }
            cursor = self.tweets.find(query).limit(limit)
            return pd.DataFrame(list(cursor))
        except Exception as e:
            logger.error(f"Error getting similar tweets: {str(e)}")
            raise DatabaseError(f"Failed to get similar tweets: {str(e)}")
    
    def get_engagement_stats(self) -> Dict[str, float]:
        """Get engagement statistics for the dataset"""
        try:
            pipeline = [
                {"$group": {
                    "_id": None,
                    "avg_engagement": {"$avg": "$engagement_score"},
                    "max_engagement": {"$max": "$engagement_score"},
                    "min_engagement": {"$min": "$engagement_score"},
                    "avg_likes": {"$avg": "$like_count"},
                    "avg_retweets": {"$avg": "$retweet_count"},
                    "avg_replies": {"$avg": "$reply_count"}
                }}
            ]
            result = list(self.tweets.aggregate(pipeline))
            return result[0] if result else {}
        except Exception as e:
            logger.error(f"Error getting engagement stats: {str(e)}")
            raise DatabaseError(f"Failed to get engagement stats: {str(e)}")
    
    def get_topic_distribution(self) -> Dict[str, int]:
        """Get distribution of topics in the dataset"""
        try:
            pipeline = [
                {"$group": {
                    "_id": "$topics",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}}
            ]
            results = list(self.tweets.aggregate(pipeline))
            
            # Parse topics from comma-separated string
            topics_dict = {}
            for result in results:
                topics = [t.strip() for t in result['_id'].split(',') if t.strip()]
                for topic in topics:
                    topics_dict[topic] = topics_dict.get(topic, 0) + result['count']
            
            return dict(sorted(topics_dict.items(), key=lambda x: x[1], reverse=True))
        except Exception as e:
            logger.error(f"Error getting topic distribution: {str(e)}")
            raise DatabaseError(f"Failed to get topic distribution: {str(e)}")
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance based on correlation with engagement"""
        try:
            # Get all relevant features
            pipeline = [
                {"$project": {
                    "text_length": 1,
                    "sentiment": 1,
                    "hashtag_count": 1,
                    "mention_count": 1,
                    "url_count": 1,
                    "has_media": 1,
                    "emoji_count": 1,
                    "word_count": 1,
                    "avg_word_length": 1,
                    "flesch_reading_ease": 1,
                    "engagement_score": 1
                }}
            ]
            cursor = self.tweets.aggregate(pipeline)
            df = pd.DataFrame(list(cursor))
            
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