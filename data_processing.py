#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data processing script for tweet analysis.
"""

import pandas as pd
import numpy as np
from database import TwitterDatabase
import logging
from datetime import datetime, timedelta
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("data_processor")

def load_data():
    """Load data from MongoDB."""
    try:
        db = TwitterDatabase()
        # Convert MongoDB documents to DataFrame
        cursor = db.tweets.find({})
        df = pd.DataFrame(list(cursor))
        logger.info(f"Loaded {len(df)} tweets from MongoDB")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def preprocess_text(text):
    """Preprocess tweet text."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    
    return " ".join(tokens)

def extract_features(df):
    """Extract features from tweet data."""
    # Preprocess text
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Calculate engagement score
    df['engagement_score'] = (
        df['retweet_count'].fillna(0) +
        df['favorite_count'].fillna(0) * 2 +
        df['reply_count'].fillna(0) * 3
    )
    
    # Extract hashtags
    df['hashtags'] = df['hashtags'].fillna('[]')
    df['hashtags'] = df['hashtags'].apply(eval)
    
    # Extract mentions
    df['mentions'] = df['mentions'].fillna('[]')
    df['mentions'] = df['mentions'].apply(eval)
    
    return df

def cluster_tweets(df, n_clusters=5):
    """Cluster tweets based on content."""
    # Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    
    # Fit and transform the text
    tfidf_matrix = vectorizer.fit_transform(df['processed_text'])
    
    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Add cluster labels to DataFrame
    df['cluster'] = clusters
    
    return df, vectorizer, kmeans

def analyze_clusters(df):
    """Analyze tweet clusters."""
    cluster_stats = []
    
    for cluster_id in df['cluster'].unique():
        cluster_df = df[df['cluster'] == cluster_id]
        
        stats = {
            'cluster_id': cluster_id,
            'size': len(cluster_df),
            'avg_engagement': cluster_df['engagement_score'].mean(),
            'top_hashtags': pd.Series([tag for tags in cluster_df['hashtags'] for tag in tags]).value_counts().head(5).to_dict(),
            'top_mentions': pd.Series([mention for mentions in cluster_df['mentions'] for mention in mentions]).value_counts().head(5).to_dict()
        }
        cluster_stats.append(stats)
    
    return cluster_stats

def update_database(df):
    """Update MongoDB with processed data."""
    try:
        db = TwitterDatabase()
        
        # Convert DataFrame to list of dictionaries
        records = df.to_dict('records')
        
        # Update documents in MongoDB
        for record in records:
            db.tweets.update_one(
                {'_id': record['_id']},
                {'$set': {
                    'processed_text': record['processed_text'],
                    'engagement_score': record['engagement_score'],
                    'cluster': record['cluster']
                }}
            )
        
        logger.info(f"Updated {len(records)} documents in MongoDB")
        
    except Exception as e:
        logger.error(f"Error updating database: {str(e)}")
        raise

def save_analysis_results(cluster_stats):
    """Save analysis results to JSON file."""
    try:
        with open('cluster_analysis.json', 'w') as f:
            json.dump(cluster_stats, f, indent=4)
        logger.info("Saved analysis results to cluster_analysis.json")
    except Exception as e:
        logger.error(f"Error saving analysis results: {str(e)}")
        raise

def main():
    """Main function to run data processing pipeline."""
    try:
        # Load data
        df = load_data()
        
        # Extract features
        df = extract_features(df)
        
        # Cluster tweets
        df, vectorizer, kmeans = cluster_tweets(df)
        
        # Analyze clusters
        cluster_stats = analyze_clusters(df)
        
        # Update database
        update_database(df)
        
        # Save results
        save_analysis_results(cluster_stats)
        
        # Log summary
        logger.info("Processing Summary:")
        logger.info(f"Total tweets processed: {len(df)}")
        logger.info(f"Number of clusters: {len(cluster_stats)}")
        logger.info(f"Average engagement score: {df['engagement_score'].mean():.2f}")
        
    except Exception as e:
        logger.error(f"Error in main processing pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main()



