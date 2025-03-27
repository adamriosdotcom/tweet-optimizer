#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tweet optimization using reinforcement learning.
"""

import pandas as pd
import numpy as np
import pickle
import random
import logging
import time
import json
import os
from database import TwitterDatabase
from process_tweets import extract_text_features, extract_topics_with_llm
from openai import OpenAI
from dotenv import load_dotenv
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from datetime import datetime
import emoji

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("reinforcement_learning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tweet_rl")

# Load environment variables
load_dotenv()

# Initialize database and OpenAI client
db = TwitterDatabase()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

class TweetEnvironment:
    """Environment for tweet optimization using RL"""
    
    def __init__(self, model_path='xgboost_model.pkl', 
                 scaler_path='feature_scaler.pkl',
                 feature_names_path='feature_names.pkl'):
        """Initialize the environment"""
        # Load the trained model and scaler
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
            
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
            
        with open(feature_names_path, 'rb') as f:
            self.feature_names = pickle.load(f)
        
        # Track top topics for one-hot encoding
        self.top_topics = self._get_top_topics(100)
        
        # Load or initialize prompt performance history
        self.prompt_history = self._load_prompt_history()
        
        logger.info("Tweet environment initialized")
        
    def _load_prompt_history(self):
        """Load prompt performance history from file"""
        try:
            with open('prompt_history.json', 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                'prompts': [],
                'performance_metrics': []
            }
    
    def _save_prompt_history(self):
        """Save prompt performance history to file"""
        try:
            with open('prompt_history.json', 'w') as f:
                json.dump(self.prompt_history, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving prompt history: {str(e)}")
    
    def _get_best_prompt(self):
        """Get the best performing prompt based on historical data"""
        if not self.prompt_history['prompts']:
            return self._get_default_prompt()
        
        # Get the prompt with the highest average performance
        best_idx = np.argmax(self.prompt_history['performance_metrics'])
        return self.prompt_history['prompts'][best_idx]
    
    def _get_default_prompt(self):
        """Get the default system prompt"""
        return """
        You are an AI tweet optimizer. Generate {n} variations of the given tweet that might improve engagement.
        Focus on:
        1. Clearer and more concise wording
        2. More engaging hooks or questions
        3. Better use of formatting (spacing, emojis, etc)
        4. More compelling calls to action
        
        Keep the same general topic and message, but vary the presentation.
        Always respect Twitter's character limit of 280 characters.
        Maintain the original author's voice and style.
        
        Format each variation as:
        1. First variation
        2. Second variation
        3. Third variation
        etc.
        """
    
    def _refine_prompt(self, current_prompt, performance_metrics):
        """Refine the prompt based on performance metrics"""
        # Add current prompt and performance to history
        self.prompt_history['prompts'].append(current_prompt)
        self.prompt_history['performance_metrics'].append(performance_metrics)
        
        # Generate a refined prompt using GPT-4o
        refinement_prompt = f"""
        Based on the following prompt and its performance metrics, suggest improvements to make it more effective.
        Current prompt:
        {current_prompt}
        
        Performance metrics:
        - Average engagement: {performance_metrics['avg_engagement']:.2f}
        - Improvement rate: {performance_metrics['improvement_rate']:.2f}%
        - Consistency score: {performance_metrics['consistency']:.2f}
        
        Suggest a refined version that:
        1. Maintains the core optimization goals
        2. Addresses any performance gaps
        3. Incorporates successful patterns from high-performing variations
        4. Keeps the same clear formatting instructions
        
        Return ONLY the refined prompt text.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a prompt engineering expert. Provide clear, specific improvements to the given prompt."},
                    {"role": "user", "content": refinement_prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            refined_prompt = response.choices[0].message.content.strip()
            
            # Validate the refined prompt
            if len(refined_prompt) > 1000:  # Basic validation
                refined_prompt = current_prompt
            
            return refined_prompt
            
        except Exception as e:
            logger.error(f"Error refining prompt: {str(e)}")
            return current_prompt
    
    def _get_top_topics(self, n=100):
        """Get top N topics from the database"""
        try:
            query = """
            SELECT topic, COUNT(*) as count
            FROM (
                SELECT trim(value) as topic
                FROM tweets
                CROSS JOIN json_each('["' || replace(topics, ',', '","') || '"]')
                WHERE topics IS NOT NULL AND topics != ''
            )
            GROUP BY topic
            ORDER BY count DESC
            LIMIT ?
            """
            
            df = pd.read_sql_query(query, db.conn, params=(n,))
            return df['topic'].tolist()
        except Exception as e:
            logger.warning(f"Failed to get top topics: {str(e)}")
            return []  # Return empty list if table doesn't exist or query fails
    
    def _tweet_to_features(self, tweet_text):
        """Convert tweet text to feature vector"""
        # Extract basic text features
        features = extract_text_features(tweet_text)
        
        # Extract topics
        topics_data = extract_topics_with_llm(tweet_text, openai_client)
        topics = topics_data.get('topics', [])
        
        # One-hot encode topics
        for topic in self.top_topics:
            features[f'topic_{topic}'] = 1 if topic in topics else 0
        
        # Create feature dataframe
        features_df = pd.DataFrame([features])
        
        # Ensure all expected features are present
        for feature in self.feature_names:
            if feature not in features_df.columns:
                features_df[feature] = 0
        
        # Select only the features used by the model
        features_df = features_df[self.feature_names]
        
        # Scale features
        features_scaled = self.scaler.transform(features_df)
        
        return features_scaled
    
    def predict_engagement(self, tweet_text):
        """Predict engagement score for a tweet"""
        features_scaled = self._tweet_to_features(tweet_text)
        
        # Predict engagement
        predicted_engagement = self.model.predict(features_scaled)[0]
        
        return predicted_engagement
    
    def generate_tweet_variations(self, seed_tweet, n_variations=5):
        """Generate variations of a seed tweet using GPT-4o"""
        # Get the best performing prompt
        system_prompt = self._get_best_prompt().format(n=n_variations)
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Original tweet: {seed_tweet}"}
                ],
                temperature=0.8,
                max_tokens=1000
            )
            
            # Extract tweet variations from response
            content = response.choices[0].message.content
            
            # Parse variations (looking for numbered lines)
            variations = []
            current_variation = ""
            
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith(tuple(f"{i}." for i in range(1, n_variations + 1))):
                    if current_variation:
                        variations.append(current_variation.strip())
                    current_variation = line.split(".", 1)[1].strip()
                elif current_variation and line:
                    current_variation += " " + line
            
            # Add the last variation
            if current_variation:
                variations.append(current_variation.strip())
            
            # If we didn't get enough variations, add some basic ones
            while len(variations) < n_variations:
                if len(variations) == 0:
                    variations.append(seed_tweet)  # Always include original as fallback
                else:
                    # Create simple variations by adding/removing punctuation or emojis
                    base = variations[0]
                    if "!" not in base:
                        variations.append(base + "!")
                    elif "?" not in base:
                        variations.append(base.replace("!", "?"))
                    else:
                        variations.append(base + " 🚀")
            
            logger.info(f"Generated {len(variations)} tweet variations")
            return variations[:n_variations]  # Ensure we don't return more than requested
            
        except Exception as e:
            logger.error(f"Error generating tweet variations: {str(e)}")
            # Return basic variations including the original
            basic_variations = [
                seed_tweet,
                seed_tweet + "!",
                seed_tweet + " 🚀",
                seed_tweet + "?",
                seed_tweet + " ✨"
            ]
            return basic_variations[:n_variations]
    
    def optimize_tweet(self, tweet_text, n_iterations=3, n_variations=5):
        """Optimize a tweet through multiple iterations"""
        current_tweet = tweet_text
        optimization_history = []
        
        # Initial engagement prediction
        initial_engagement = self.predict_engagement(current_tweet)
        optimization_history.append({
            'iteration': 0,
            'tweet': current_tweet,
            'predicted_engagement': initial_engagement,
            'is_exploration': False
        })
        
        logger.info(f"Starting optimization with seed tweet. Initial predicted engagement: {initial_engagement:.4f}")
        
        # Enhanced performance metrics tracking
        performance_metrics = {
            'avg_engagement': 0,
            'improvement_rate': 0,
            'consistency': 0,
            'max_engagement': 0,
            'min_engagement': 0,
            'engagement_range': 0,
            'top_percentile': 0,
            'bottom_percentile': 0,
            'iteration_count': n_iterations,
            'variation_count': n_variations,
            'timestamp': datetime.now().isoformat(),
            'original_length': len(tweet_text),
            'avg_length': 0,
            'hashtag_ratio': 0,
            'emoji_ratio': 0,
            'question_ratio': 0,
            'exclamation_ratio': 0,
            'topic_coverage': 0
        }
        
        # Iterative optimization
        for i in range(n_iterations):
            # Generate variations
            variations = self.generate_tweet_variations(current_tweet, n_variations)
            
            # Predict engagement for each variation
            engagement_scores = [self.predict_engagement(tweet) for tweet in variations]
            
            # Select best variation
            best_idx = np.argmax(engagement_scores)
            best_tweet = variations[best_idx]
            predicted_engagement = engagement_scores[best_idx]
            
            # Update performance metrics
            performance_metrics.update({
                'avg_engagement': np.mean(engagement_scores),
                'max_engagement': np.max(engagement_scores),
                'min_engagement': np.min(engagement_scores),
                'engagement_range': np.max(engagement_scores) - np.min(engagement_scores),
                'top_percentile': np.percentile(engagement_scores, 90),
                'bottom_percentile': np.percentile(engagement_scores, 10),
                'improvement_rate': ((predicted_engagement / initial_engagement) - 1) * 100,
                'consistency': 1 - (np.std(engagement_scores) / np.mean(engagement_scores)),
                'avg_length': np.mean([len(tweet) for tweet in variations]),
                'hashtag_ratio': np.mean([tweet.count('#') for tweet in variations]),
                'emoji_ratio': np.mean([sum(1 for char in tweet if char in emoji.EMOJI_DATA) for tweet in variations]),
                'question_ratio': np.mean([tweet.count('?') for tweet in variations]),
                'exclamation_ratio': np.mean([tweet.count('!') for tweet in variations])
            })
            
            # Calculate topic coverage
            all_topics = []
            for tweet in variations:
                topics_data = extract_topics_with_llm(tweet, openai_client)
                all_topics.extend(topics_data.get('topics', []))
            performance_metrics['topic_coverage'] = len(set(all_topics)) / len(self.top_topics)
            
            # Record history
            optimization_history.append({
                'iteration': i + 1,
                'tweet': best_tweet,
                'predicted_engagement': predicted_engagement,
                'variations': variations,
                'selected_idx': best_idx,
                'is_exploration': False,
                'performance_metrics': performance_metrics.copy()  # Store metrics for each iteration
            })
            
            logger.info(f"Iteration {i+1}: Selected tweet with predicted engagement {predicted_engagement:.4f}")
            
            # Update current tweet for next iteration
            current_tweet = best_tweet
        
        # Refine the prompt based on performance
        current_prompt = self._get_best_prompt()
        refined_prompt = self._refine_prompt(current_prompt, performance_metrics)
        
        # Save the refined prompt if it's different
        if refined_prompt != current_prompt:
            self.prompt_history['prompts'].append(refined_prompt)
            self.prompt_history['performance_metrics'].append(performance_metrics)
            self._save_prompt_history()
        
        # Return the final optimized tweet and the optimization history
        return current_tweet, optimization_history

class TweetOptimizer:
    """Optimize tweets using RL techniques"""
    
    def __init__(self, environment):
        """Initialize the optimizer"""
        self.env = environment
        self.history = []
        self.epsilon = 0.1  # Exploration rate
        
    def select_best_tweet(self, tweet_variations, exploration=True):
        """Select the best tweet from variations using epsilon-greedy"""
        if exploration and random.random() < self.epsilon:
            # Exploration: select random variation
            selected_idx = random.randint(0, len(tweet_variations) - 1)
            selected_tweet = tweet_variations[selected_idx]
            logger.info(f"Exploration: selected random variation {selected_idx}")
            is_exploration = True
        else:
            # Exploitation: select highest predicted engagement
            engagement_scores = [self.env.predict_engagement(tweet) for tweet in tweet_variations]
            selected_idx = np.argmax(engagement_scores)
            selected_tweet = tweet_variations[selected_idx]
            logger.info(f"Exploitation: selected variation {selected_idx} with predicted engagement {engagement_scores[selected_idx]:.4f}")
            is_exploration = False
        
        return selected_tweet, selected_idx, is_exploration
    
    def optimize_tweet(self, seed_tweet, n_iterations=3, n_variations=5):
        """Iteratively optimize a tweet through multiple generations"""
        current_tweet = seed_tweet
        optimization_history = []
        
        # Initial engagement prediction
        initial_engagement = self.env.predict_engagement(current_tweet)
        optimization_history.append({
            'iteration': 0,
            'tweet': current_tweet,
            'predicted_engagement': initial_engagement,
            'is_exploration': False
        })
        
        logger.info(f"Starting optimization with seed tweet. Initial predicted engagement: {initial_engagement:.4f}")
        
        # Iterative optimization
        for i in range(n_iterations):
            # Generate variations
            variations = self.env.generate_tweet_variations(current_tweet, n_variations)
            
            # Select best variation
            best_tweet, selected_idx, is_exploration = self.select_best_tweet(variations)
            predicted_engagement = self.env.predict_engagement(best_tweet)
            
            # Record history
            optimization_history.append({
                'iteration': i + 1,
                'tweet': best_tweet,
                'predicted_engagement': predicted_engagement,
                'variations': variations,
                'selected_idx': selected_idx,
                'is_exploration': is_exploration
            })
            
            logger.info(f"Iteration {i+1}: Selected tweet with predicted engagement {predicted_engagement:.4f}")
            
            # Update current tweet for next iteration
            current_tweet = best_tweet
        
        # Return the final optimized tweet and the optimization history
        return current_tweet, optimization_history
    
    def update_policy(self, actual_engagement):
        """Update policy based on observed engagement"""
        # Simple update: adjust epsilon based on performance
        # This is a placeholder for more sophisticated policy updates
        if len(self.history) > 0:
            last_predicted = self.history[-1]['predicted_engagement']
            
            # If actual is better than predicted, reduce exploration (more confidence)
            # If actual is worse than predicted, increase exploration (less confidence)
            ratio = actual_engagement / max(last_predicted, 0.001)
            
            if ratio > 1.2:  # Actual much better than predicted
                self.epsilon = max(0.05, self.epsilon * 0.9)
                logger.info(f"Reducing exploration rate to {self.epsilon:.4f}")
            elif ratio < 0.8:  # Actual much worse than predicted
                self.epsilon = min(0.5, self.epsilon * 1.1)
                logger.info(f"Increasing exploration rate to {self.epsilon:.4f}")

def main():
    """Main function for tweet optimization demo"""
    # Initialize environment and optimizer
    env = TweetEnvironment()
    optimizer = TweetOptimizer(env)
    
    # Example seed tweet
    seed_tweet = "Just released our new AI model! It's faster and more accurate than previous versions. Check it out at our website."
    
    # Optimize the tweet
    optimized_tweet, history = optimizer.optimize_tweet(seed_tweet)
    
    # Print results
    logger.info("\nOptimization Results:")
    logger.info(f"Seed Tweet: {seed_tweet}")
    logger.info(f"Predicted Engagement: {history[0]['predicted_engagement']:.4f}")
    logger.info(f"\nOptimized Tweet: {optimized_tweet}")
    logger.info(f"Predicted Engagement: {history[-1]['predicted_engagement']:.4f}")
    logger.info(f"Improvement: {(history[-1]['predicted_engagement'] / history[0]['predicted_engagement'] - 1) * 100:.2f}%")
    
    # Save history to file for analysis
    with open('optimization_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("Optimization history saved to optimization_history.json")

if __name__ == "__main__":
    main() 