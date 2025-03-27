#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Feedback collection system for reinforcement learning.
"""

import pandas as pd
import numpy as np
import logging
import time
from datetime import datetime, timedelta
import json
from database import TwitterDatabase
from tweet_rl import TweetOptimizer, TweetEnvironment

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedback_collector.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feedback_collector")

# Initialize database
db = TwitterDatabase()

class FeedbackCollector:
    """Collects feedback on tweet performance for RL using historical data"""
    
    def __init__(self, db, env, optimizer, feedback_window_days=30):
        """Initialize the feedback collector"""
        self.db = db
        self.env = env
        self.optimizer = optimizer
        self.feedback_window = feedback_window_days
        
        # Load optimization history if exists
        self.optimization_history = self._load_optimization_history()
        
        logger.info(f"Feedback collector initialized with {len(self.optimization_history)} historical optimizations")
    
    def _load_optimization_history(self):
        """Load optimization history from file"""
        try:
            with open('optimization_history.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return []
    
    def save_optimization_history(self):
        """Save optimization history to file"""
        try:
            # Convert numpy types to Python native types
            history = []
            for entry in self.optimization_history:
                entry_copy = entry.copy()
                for key, value in entry_copy.items():
                    if hasattr(value, 'item'):  # Check if it's a numpy type
                        entry_copy[key] = value.item()
                history.append(entry_copy)
            
            with open('optimization_history.json', 'w') as f:
                json.dump(history, f, indent=2)
            
            logger.info("Optimization history saved successfully")
        except Exception as e:
            logger.error(f"Error saving optimization history: {str(e)}")
            raise DatabaseError(f"Failed to save optimization history: {str(e)}")
    
    def collect_feedback(self, limit=100):
        """Collect feedback for recent optimized tweets using historical data"""
        # Get tweets that were optimized by this system
        optimized_tweets = [entry for entry in self.optimization_history 
                          if 'actual_engagement' not in entry]
        
        # Limit to most recent ones
        optimized_tweets = optimized_tweets[-limit:]
        
        if not optimized_tweets:
            logger.info("No optimized tweets to collect feedback for")
            return 0
        
        logger.info(f"Collecting feedback for {len(optimized_tweets)} optimized tweets")
        
        feedback_count = 0
        
        for tweet_entry in optimized_tweets:
            # Check if we have the actual tweet in our database
            tweet_text = tweet_entry.get('tweet')
            
            if not tweet_text:
                continue
            
            # Query database for similar tweets
            query = """
            SELECT 
                uniqueid, text, like_count, retweet_count, reply_count,
                quote_count, view_count, bookmark_count
            FROM tweets
            WHERE text LIKE ?
            LIMIT 1
            """
            
            tweet_data = pd.read_sql_query(query, self.db.conn, params=(f"%{tweet_text[:50]}%",))
            
            if len(tweet_data) > 0:
                # Calculate actual engagement from historical data
                row = tweet_data.iloc[0]
                actual_engagement = (
                    row['like_count'] * 1.0 +
                    row['retweet_count'] * 2.0 +
                    row['reply_count'] * 1.5 +
                    row['quote_count'] * 1.8 +
                    row['bookmark_count'] * 1.2
                )
                
                # Update history with actual engagement
                tweet_entry['actual_engagement'] = float(actual_engagement)
                tweet_entry['feedback_collected_at'] = datetime.now().isoformat()
                
                # Update the optimizer with this feedback
                self.optimizer.update_policy(actual_engagement)
                
                feedback_count += 1
                
                logger.info(f"Collected feedback for tweet. Predicted: {tweet_entry['predicted_engagement']:.4f}, Actual: {actual_engagement:.4f}")
        
        # Save updated history
        self.save_optimization_history()
        
        return feedback_count
    
    def analyze_performance(self):
        """Analyze performance of predictions vs actual engagements"""
        # Get entries with both predicted and actual engagement
        complete_entries = [entry for entry in self.optimization_history 
                          if 'actual_engagement' in entry]
        
        if not complete_entries:
            logger.info("No complete entries for performance analysis")
            return None
        
        # Create dataframe with predicted and actual values
        df = pd.DataFrame({
            'predicted': [entry['predicted_engagement'] for entry in complete_entries],
            'actual': [entry['actual_engagement'] for entry in complete_entries],
            'is_exploration': [entry.get('is_exploration', False) for entry in complete_entries],
            'timestamp': [entry.get('feedback_collected_at', '') for entry in complete_entries]
        })
        
        # Calculate error metrics
        df['error'] = df['predicted'] - df['actual']
        df['abs_error'] = abs(df['error'])
        df['relative_error'] = df['error'] / df['actual'].replace(0, 0.001)
        
        # Overall metrics
        mean_abs_error = df['abs_error'].mean()
        mean_rel_error = df['relative_error'].abs().mean()
        
        # Exploration vs exploitation comparison
        exploration_error = df[df['is_exploration']]['abs_error'].mean()
        exploitation_error = df[~df['is_exploration']]['abs_error'].mean()
        
        # Output results
        logger.info("\nPerformance Analysis:")
        logger.info(f"Number of entries: {len(df)}")
        logger.info(f"Mean Absolute Error: {mean_abs_error:.4f}")
        logger.info(f"Mean Relative Error: {mean_rel_error:.4f}")
        logger.info(f"Exploration Mean Abs Error: {exploration_error:.4f}")
        logger.info(f"Exploitation Mean Abs Error: {exploitation_error:.4f}")
        
        return {
            'mean_absolute_error': float(mean_abs_error),
            'mean_relative_error': float(mean_rel_error),
            'exploration_error': float(exploration_error),
            'exploitation_error': float(exploitation_error),
            'sample_size': len(df)
        }

def main():
    """Main function for feedback collection"""
    # Initialize environment and optimizer
    env = TweetEnvironment()
    optimizer = TweetOptimizer(env)
    
    # Initialize database
    db = TwitterDatabase()
    
    # Initialize feedback collector
    collector = FeedbackCollector(db, env, optimizer)
    
    # Collect feedback
    feedback_count = collector.collect_feedback()
    logger.info(f"Collected feedback for {feedback_count} tweets")
    
    # Analyze performance
    performance = collector.analyze_performance()
    
    if performance:
        # Save performance metrics
        with open('performance_metrics.json', 'w') as f:
            json.dump(performance, f, indent=2)
        
        logger.info("Performance metrics saved to performance_metrics.json")

if __name__ == "__main__":
    main() 