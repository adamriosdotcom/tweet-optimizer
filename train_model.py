#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train XGBoost model for tweet engagement prediction.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_trainer")

def get_available_columns():
    """Get list of available columns in the tweets table."""
    conn = sqlite3.connect('tweets.db')
    query = "SELECT name FROM pragma_table_info('tweets')"
    columns = pd.read_sql_query(query, conn)['name'].tolist()
    conn.close()
    return columns

def remove_outliers(df, columns, z_threshold=3):
    """Remove outliers using z-score method."""
    df_clean = df.copy()
    for col in columns:
        z_scores = stats.zscore(df_clean[col])
        df_clean = df_clean[abs(z_scores) < z_threshold]
    return df_clean

def load_processed_tweets():
    """Load processed tweets from the database."""
    conn = sqlite3.connect('tweets.db')
    
    # Get available columns
    available_columns = get_available_columns()
    logger.info(f"Available columns: {available_columns}")
    
    # Define feature columns based on what's available
    feature_columns = [
        col for col in [
            # Text features
            'text_length', 'word_count', 'avg_word_length',
            'hashtag_count', 'mention_count', 'url_count', 'emoji_count',
            'sentiment', 'flesch_reading_ease', 'subjectivity',
            'tweet_complexity', 'punctuation_count', 'uppercase_ratio',
            
            # Author features
            'daily_author_activity', 'author_tweet_count',
            'tweets_last_7_days', 'tweets_last_30_days',
            'avg_tweets_per_day_7d', 'avg_tweets_per_day_30d',
            'followers_per_status', 'follower_growth_rate',
            'popularity_index', 'account_age_days',
            
            # Engagement features
            'likes_per_follower', 'rolling_engagement',
            'retweet_like_ratio', 'quote_influence',
            'response_time_minutes', 'verified_engagement_avg',
            
            # Content features
            'has_media', 'has_location', 'media_aspect_ratio',
            'bio_length', 'bio_url_count',
            
            # Language and topics
            'language', 'topic'
        ] if col in available_columns
    ]
    
    logger.info(f"Using feature columns: {feature_columns}")
    
    # Load data
    query = f"""
    SELECT {', '.join(feature_columns)}, 
           (likeCount + retweetCount + replyCount + quoteCount) as engagement_score
    FROM tweets
    WHERE text_length IS NOT NULL AND sentiment IS NOT NULL
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Remove outliers from engagement score
    df = remove_outliers(df, ['engagement_score'])
    
    return df, feature_columns

def prepare_features(df, feature_columns):
    """Prepare features for model training."""
    # Separate numerical and categorical features
    numerical_features = [
        col for col in feature_columns
        if col not in ['language', 'topic']
    ]
    
    categorical_features = [
        col for col in ['language', 'topic']
        if col in feature_columns
    ]
    
    logger.info(f"Numerical features: {numerical_features}")
    logger.info(f"Categorical features: {categorical_features}")
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_features)
        ])
    
    # Prepare target variable
    y = df['engagement_score']
    
    # Create feature matrix
    X = df[numerical_features + categorical_features]
    
    return X, y, preprocessor

def train_model():
    """Train the XGBoost model."""
    try:
        logger.info("Loading processed tweets...")
        df, feature_columns = load_processed_tweets()
        
        logger.info("Preparing features...")
        X, y, preprocessor = prepare_features(df, feature_columns)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create pipeline with more complex model
        model = Pipeline([
            ('preprocessor', preprocessor),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=200,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=3,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            ))
        ])
        
        logger.info("Training model...")
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Model Performance:")
        logger.info(f"MSE: {mse:.2f}")
        logger.info(f"R² Score: {r2:.2f}")
        
        # Save the model and preprocessor
        logger.info("Saving model and preprocessor...")
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)
        
        logger.info("Model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise

if __name__ == "__main__":
    train_model() 