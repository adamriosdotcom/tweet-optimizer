#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train the RL model using historical tweet data.
"""

import pandas as pd
import numpy as np
from tweet_rl import TweetEnvironment
from database import TwitterDatabase
import logging
from datetime import datetime, timedelta
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("rl_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("rl_training")

def load_training_data(db, limit=1000):
    """Load historical tweet data for training"""
    query = """
    SELECT text, engagement_score, topics, created_at
    FROM tweets
    WHERE engagement_score IS NOT NULL
    ORDER BY created_at DESC
    LIMIT ?
    """
    
    df = pd.read_sql_query(query, db.conn, params=(limit,))
    logger.info(f"Loaded {len(df)} tweets for training")
    return df

def simulate_optimization(env, tweet_text, true_engagement):
    """Simulate tweet optimization without posting"""
    # Get initial prediction
    initial_features = env._tweet_to_features(tweet_text)
    initial_prediction = env.model.predict(initial_features)[0]
    
    # Optimize tweet
    optimized_text, history = env.optimize_tweet(tweet_text)
    optimized_features = env._tweet_to_features(optimized_text)
    optimized_prediction = env.model.predict(optimized_features)[0]
    
    # Calculate improvement
    improvement = (optimized_prediction - initial_prediction) / initial_prediction * 100
    
    return {
        'original_text': tweet_text,
        'optimized_text': optimized_text,
        'initial_prediction': initial_prediction,
        'optimized_prediction': optimized_prediction,
        'true_engagement': true_engagement,
        'improvement': improvement
    }

def train_model(episodes=100, batch_size=50):
    """Train the RL model using historical data"""
    # Initialize environment and database
    env = TweetEnvironment()
    db = TwitterDatabase()
    
    # Load training data
    training_data = load_training_data(db)
    
    # Training metrics
    total_improvements = []
    episode_rewards = []
    
    logger.info(f"Starting training with {episodes} episodes")
    
    for episode in tqdm(range(episodes), desc="Training episodes"):
        episode_reward = 0
        episode_improvements = []
        
        # Sample batch of tweets
        batch = training_data.sample(n=min(batch_size, len(training_data)))
        
        for _, tweet in batch.iterrows():
            # Simulate optimization
            result = simulate_optimization(env, tweet['text'], tweet['engagement_score'])
            
            # Calculate reward based on improvement
            reward = result['improvement']
            episode_reward += reward
            episode_improvements.append(result['improvement'])
            
            # Log significant improvements
            if result['improvement'] > 10:
                logger.info(f"Significant improvement: {result['improvement']:.2f}%")
                logger.info(f"Original: {result['original_text']}")
                logger.info(f"Optimized: {result['optimized_text']}")
        
        # Record episode metrics
        avg_improvement = np.mean(episode_improvements)
        total_improvements.append(avg_improvement)
        episode_rewards.append(episode_reward)
        
        # Log episode summary
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode + 1}/{episodes}")
            logger.info(f"Average improvement: {avg_improvement:.2f}%")
            logger.info(f"Episode reward: {episode_reward:.2f}")
    
    # Training summary
    logger.info("\nTraining completed!")
    logger.info(f"Average improvement across all episodes: {np.mean(total_improvements):.2f}%")
    logger.info(f"Best improvement: {max(total_improvements):.2f}%")
    logger.info(f"Average episode reward: {np.mean(episode_rewards):.2f}")
    
    return total_improvements, episode_rewards

if __name__ == "__main__":
    # Train the model
    improvements, rewards = train_model(episodes=100, batch_size=50) 