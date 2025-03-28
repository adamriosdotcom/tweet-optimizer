#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model training script for tweet analysis.
"""

import pandas as pd
import numpy as np
from database import TwitterDatabase
import logging
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("model_trainer")

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

def prepare_features(df):
    """Prepare features for model training."""
    # Calculate engagement score if not exists
    if 'engagement_score' not in df.columns:
        df['engagement_score'] = (
            df['retweet_count'].fillna(0) +
            df['favorite_count'].fillna(0) * 2 +
            df['reply_count'].fillna(0) * 3
        )
    
    # Create binary target variable (high engagement vs low engagement)
    df['is_high_engagement'] = (df['engagement_score'] > df['engagement_score'].median()).astype(int)
    
    # Prepare text features
    df['processed_text'] = df['text'].fillna('')
    
    return df

def train_model(df):
    """Train the model."""
    # Split features and target
    X = df['processed_text']
    y = df['is_high_engagement']
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train Random Forest model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test_tfidf)
    report = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred)
    
    # Save model and vectorizer
    joblib.dump(model, 'tweet_model.joblib')
    joblib.dump(vectorizer, 'tweet_vectorizer.joblib')
    
    return {
        'classification_report': report,
        'confusion_matrix': confusion.tolist(),
        'feature_importance': dict(zip(
            vectorizer.get_feature_names_out(),
            model.feature_importances_
        ))
    }

def save_results(results):
    """Save training results to JSON file."""
    try:
        with open('model_results.json', 'w') as f:
            json.dump(results, f, indent=4)
        logger.info("Saved model results to model_results.json")
    except Exception as e:
        logger.error(f"Error saving model results: {str(e)}")
        raise

def main():
    """Main function to run model training pipeline."""
    try:
        # Load data
        df = load_data()
        
        # Prepare features
        df = prepare_features(df)
        
        # Train model
        results = train_model(df)
        
        # Save results
        save_results(results)
        
        # Log summary
        logger.info("Training Summary:")
        logger.info(f"Total samples: {len(df)}")
        logger.info(f"High engagement samples: {df['is_high_engagement'].sum()}")
        logger.info("\nClassification Report:")
        logger.info(results['classification_report'])
        
    except Exception as e:
        logger.error(f"Error in main training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    main() 