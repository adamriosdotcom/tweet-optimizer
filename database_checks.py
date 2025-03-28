#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Database health check script for MongoDB.
"""

import logging
from database import TwitterDatabase
from datetime import datetime, timedelta
import pandas as pd

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("database_checks.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("database_checker")

def check_database_health():
    """Check the health of the MongoDB database."""
    try:
        db = TwitterDatabase()
        
        # Get collection stats
        stats = db.tweets.count_documents({})
        logger.info(f"Total documents in collection: {stats}")
        
        # Check for recent tweets
        recent_date = datetime.utcnow() - timedelta(days=7)
        recent_count = db.tweets.count_documents({
            "created_at": {"$gte": recent_date}
        })
        logger.info(f"Tweets from last 7 days: {recent_count}")
        
        # Check for missing required fields
        missing_fields = {
            "text": db.tweets.count_documents({"text": {"$exists": False}}),
            "created_at": db.tweets.count_documents({"created_at": {"$exists": False}}),
            "author": db.tweets.count_documents({"author": {"$exists": False}}),
            "engagement_score": db.tweets.count_documents({"engagement_score": {"$exists": False}})
        }
        
        logger.info("Missing fields count:")
        for field, count in missing_fields.items():
            logger.info(f"{field}: {count}")
        
        # Check for duplicate tweets
        pipeline = [
            {"$group": {
                "_id": "$id",
                "count": {"$sum": 1}
            }},
            {"$match": {
                "count": {"$gt": 1}
            }}
        ]
        duplicates = list(db.tweets.aggregate(pipeline))
        logger.info(f"Number of duplicate tweets: {len(duplicates)}")
        
        # Check data types
        sample_doc = db.tweets.find_one()
        if sample_doc:
            logger.info("Sample document fields and types:")
            for field, value in sample_doc.items():
                logger.info(f"{field}: {type(value)}")
        
        # Check indexes
        indexes = db.tweets.list_indexes()
        logger.info("Collection indexes:")
        for index in indexes:
            logger.info(f"Index: {index}")
        
        return {
            "total_documents": stats,
            "recent_tweets": recent_count,
            "missing_fields": missing_fields,
            "duplicate_count": len(duplicates),
            "indexes": list(indexes)
        }
        
    except Exception as e:
        logger.error(f"Error checking database health: {str(e)}")
        raise

def check_data_quality():
    """Check the quality of the data in the database."""
    try:
        db = TwitterDatabase()
        
        # Convert to DataFrame for analysis
        cursor = db.tweets.find({})
        df = pd.DataFrame(list(cursor))
        
        # Check for null values
        null_counts = df.isnull().sum()
        logger.info("Null value counts:")
        for column, count in null_counts.items():
            if count > 0:
                logger.info(f"{column}: {count}")
        
        # Check for outliers in engagement scores
        if 'engagement_score' in df.columns:
            q1 = df['engagement_score'].quantile(0.25)
            q3 = df['engagement_score'].quantile(0.75)
            iqr = q3 - q1
            outliers = df[
                (df['engagement_score'] < q1 - 1.5 * iqr) |
                (df['engagement_score'] > q3 + 1.5 * iqr)
            ]
            logger.info(f"Number of engagement score outliers: {len(outliers)}")
        
        # Check text length distribution
        if 'text' in df.columns:
            df['text_length'] = df['text'].str.len()
            logger.info("Text length statistics:")
            logger.info(df['text_length'].describe())
        
        # Check date range
        if 'created_at' in df.columns:
            df['created_at'] = pd.to_datetime(df['created_at'])
            logger.info("Date range:")
            logger.info(f"Earliest tweet: {df['created_at'].min()}")
            logger.info(f"Latest tweet: {df['created_at'].max()}")
        
        return {
            "null_counts": null_counts.to_dict(),
            "engagement_outliers": len(outliers) if 'engagement_score' in df.columns else 0,
            "text_length_stats": df['text_length'].describe().to_dict() if 'text' in df.columns else {},
            "date_range": {
                "earliest": str(df['created_at'].min()) if 'created_at' in df.columns else None,
                "latest": str(df['created_at'].max()) if 'created_at' in df.columns else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error checking data quality: {str(e)}")
        raise

def main():
    """Main function to run database checks."""
    try:
        # Check database health
        health_results = check_database_health()
        logger.info("\nDatabase Health Results:")
        logger.info(health_results)
        
        # Check data quality
        quality_results = check_data_quality()
        logger.info("\nData Quality Results:")
        logger.info(quality_results)
        
    except Exception as e:
        logger.error(f"Error in main database check: {str(e)}")
        raise

if __name__ == "__main__":
    main()