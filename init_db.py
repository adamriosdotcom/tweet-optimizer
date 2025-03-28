#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Initialize the MongoDB database with the proper schema and indexes.
"""

import logging
from database import TwitterDatabase
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("init_db.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("db_initializer")

def create_indexes(db):
    """Create necessary indexes for the tweets collection."""
    try:
        # Create indexes for commonly queried fields
        indexes = [
            [("created_at", 1)],
            [("author.userName", 1)],
            [("engagement_score", -1)],
            [("processed", 1)],
            [("error", 1)],
            [("topics", 1)],
            [("uniqueid", 1), {"unique": True}]
        ]
        
        for index in indexes:
            db.tweets.create_index(index)
            logger.info(f"Created index: {index}")
        
        logger.info("All indexes created successfully")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {str(e)}")
        raise

def validate_schema(db):
    """Validate the database schema."""
    try:
        # Check if collection exists
        if "tweets" not in db.list_collection_names():
            logger.info("Creating tweets collection")
            db.create_collection("tweets")
        
        # Get collection stats
        stats = db.tweets.count_documents({})
        logger.info(f"Current document count: {stats}")
        
        # Check indexes
        indexes = list(db.tweets.list_indexes())
        logger.info(f"Current indexes: {len(indexes)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error validating schema: {str(e)}")
        return False

def main():
    """Main function to initialize the database."""
    try:
        logger.info("Initializing database...")
        
        # Initialize database connection
        db = TwitterDatabase()
        
        # Validate schema
        if not validate_schema(db):
            raise Exception("Schema validation failed")
        
        # Create indexes
        create_indexes(db)
        
        logger.info("Database initialization completed successfully")
        
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        raise

if __name__ == "__main__":
    main() 