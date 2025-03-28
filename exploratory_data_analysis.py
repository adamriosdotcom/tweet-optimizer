#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exploratory Data Analysis script for tweet data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from database import TwitterDatabase
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("eda.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("eda")

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

def analyze_engagement(df):
    """Analyze engagement metrics."""
    # Calculate engagement score
    df['engagement_score'] = (
        df['retweet_count'].fillna(0) +
        df['favorite_count'].fillna(0) * 2 +
        df['reply_count'].fillna(0) * 3
    )
    
    # Basic statistics
    stats = {
        'total_tweets': len(df),
        'avg_engagement': df['engagement_score'].mean(),
        'max_engagement': df['engagement_score'].max(),
        'min_engagement': df['engagement_score'].min(),
        'std_engagement': df['engagement_score'].std()
    }
    
    # Engagement distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='engagement_score', bins=50)
    plt.title('Distribution of Engagement Scores')
    plt.xlabel('Engagement Score')
    plt.ylabel('Count')
    plt.savefig('engagement_distribution.png')
    plt.close()
    
    return stats

def analyze_temporal_patterns(df):
    """Analyze temporal patterns in tweet posting."""
    # Convert created_at to datetime
    df['created_at'] = pd.to_datetime(df['created_at'])
    
    # Hourly distribution
    df['hour'] = df['created_at'].dt.hour
    hourly_counts = df.groupby('hour').size()
    
    plt.figure(figsize=(12, 6))
    hourly_counts.plot(kind='bar')
    plt.title('Tweet Distribution by Hour')
    plt.xlabel('Hour of Day')
    plt.ylabel('Number of Tweets')
    plt.savefig('hourly_distribution.png')
    plt.close()
    
    # Daily distribution
    df['date'] = df['created_at'].dt.date
    daily_counts = df.groupby('date').size()
    
    plt.figure(figsize=(12, 6))
    daily_counts.plot(kind='line')
    plt.title('Tweet Distribution by Date')
    plt.xlabel('Date')
    plt.ylabel('Number of Tweets')
    plt.savefig('daily_distribution.png')
    plt.close()

def analyze_topics(df):
    """Analyze topics and hashtags."""
    # Extract hashtags
    df['hashtags'] = df['hashtags'].fillna('[]')
    df['hashtags'] = df['hashtags'].apply(eval)
    
    # Flatten hashtags list
    all_hashtags = [tag for tags in df['hashtags'] for tag in tags]
    
    # Count hashtag frequencies
    hashtag_counts = pd.Series(all_hashtags).value_counts().head(20)
    
    plt.figure(figsize=(12, 6))
    hashtag_counts.plot(kind='bar')
    plt.title('Top 20 Hashtags')
    plt.xlabel('Hashtag')
    plt.ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('hashtag_distribution.png')
    plt.close()

def analyze_user_metrics(df):
    """Analyze user-related metrics."""
    # User tweet counts
    user_counts = df['user_screen_name'].value_counts().head(20)
    
    plt.figure(figsize=(12, 6))
    user_counts.plot(kind='bar')
    plt.title('Top 20 Users by Tweet Count')
    plt.xlabel('Username')
    plt.ylabel('Number of Tweets')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('user_distribution.png')
    plt.close()
    
    # User engagement
    user_engagement = df.groupby('user_screen_name')['engagement_score'].mean().sort_values(ascending=False).head(20)
    
    plt.figure(figsize=(12, 6))
    user_engagement.plot(kind='bar')
    plt.title('Top 20 Users by Average Engagement')
    plt.xlabel('Username')
    plt.ylabel('Average Engagement Score')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('user_engagement.png')
    plt.close()

def main():
    """Main function to run all analyses."""
    try:
        # Load data
        df = load_data()
        
        # Run analyses
        engagement_stats = analyze_engagement(df)
        analyze_temporal_patterns(df)
        analyze_topics(df)
        analyze_user_metrics(df)
        
        # Log results
        logger.info("Analysis Results:")
        logger.info(f"Total Tweets: {engagement_stats['total_tweets']}")
        logger.info(f"Average Engagement: {engagement_stats['avg_engagement']:.2f}")
        logger.info(f"Max Engagement: {engagement_stats['max_engagement']:.2f}")
        logger.info(f"Min Engagement: {engagement_stats['min_engagement']:.2f}")
        logger.info(f"Engagement Std Dev: {engagement_stats['std_engagement']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()
