#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 08:56:25 2025

@author: adamrios
"""

from apify_client import ApifyClient
import pandas as pd
import numpy as np
import datetime
import logging
from database import TwitterDatabase
from dotenv import load_dotenv
import os

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("scraper.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tweet_scraper")

# Load environment variables
load_dotenv()

# Your Apify API token
APIFY_TOKEN = os.getenv('APIFY_API_KEY')

# Initialize the Apify client
client = ApifyClient(APIFY_TOKEN)

def scrape_user_tweets(username, max_tweets=100):
    """Scrape tweets from a specific user."""
    try:
        input_data = {
            "filter:blue_verified": False,
            "filter:consumer_video": False,
            "filter:has_engagement": False,
            "filter:hashtags": False,
            "filter:images": False,
            "filter:links": False,
            "filter:media": False,
            "filter:mentions": False,
            "filter:native_video": False,
            "filter:nativeretweets": False,
            "filter:news": False,
            "filter:pro_video": False,
            "filter:quote": False,
            "filter:replies": False,
            "filter:safe": False,
            "filter:spaces": False,
            "filter:twimg": False,
            "filter:verified": False,
            "filter:videos": False,
            "filter:vine": False,
            "from": username,
            "include:nativeretweets": False,
            "lang": "en",
            "queryType": "Latest",
            "maxItems": max_tweets
        }
        
        # Run the Twitter Scraper actor
        posts_actor_id = "kaitoeasyapi/twitter-x-data-tweet-scraper-pay-per-result-cheapest"
        posts_run = client.actor(posts_actor_id).call(run_input=input_data)
        
        # Get dataset ID and fetch items
        posts_dataset_id = posts_run["defaultDatasetId"]
        posts_results = client.dataset(posts_dataset_id).list_items().items
        
        valid_posts = [post for post in posts_results if post['type'] == 'tweet']
        
        if valid_posts:
            dfs = [pd.json_normalize(inner_list) for inner_list in valid_posts if inner_list]
            if dfs:
                tweets = pd.concat(dfs, ignore_index=True)
                
                # Add datetime and uniqueid
                tweets['datetime'] = pd.to_datetime(str(np.datetime64(posts_run['finishedAt'].replace(tzinfo=None)))).strftime("%Y%m%d%H%M%S")
                tweets['uniqueid'] = tweets['id'] + "_" + tweets['datetime']
                
                # Process list columns
                list_columns = [col for col in tweets.columns if tweets[col].apply(lambda x: isinstance(x, list)).any()]
                for col in list_columns:
                    tweets[col] = tweets[col].apply(lambda x: ','.join(map(str, x)) if isinstance(x, list) else x)
                
                return tweets
            else:
                logger.warning(f"No valid posts to concatenate for {username}")
                return None
        else:
            logger.warning(f"No valid tweets found for {username}")
            return None
            
    except Exception as e:
        logger.error(f"Error scraping tweets for {username}: {str(e)}")
        return None

def save_to_mongodb(tweets_df):
    """Save tweets to MongoDB."""
    try:
        db = TwitterDatabase()
        
        # Convert DataFrame to list of dictionaries
        records = tweets_df.to_dict('records')
        
        # Insert into MongoDB
        result = db.tweets.insert_many(records, ordered=False)
        logger.info(f"Added {len(result.inserted_ids)} tweets to MongoDB")
        return len(result.inserted_ids)
        
    except Exception as e:
        logger.error(f"Error saving to MongoDB: {str(e)}")
        raise

def scrape_following_list(username, max_users=50):
    """Scrape tweets from users in a following list."""
    try:
        get_users = {
            "getFollowers": False,
            "getFollowing": True,
            "maxItems": max_users,
            "twitterHandles": [username]
        }
        
        following_actor_id = "apidojo/twitter-user-scraper"
        following_run = client.actor(following_actor_id).call(run_input=get_users)
        
        # Get the dataset ID for the results
        following_dataset_id = following_run["defaultDatasetId"]
        
        # Fetch the results from the dataset
        following_results = client.dataset(following_dataset_id).list_items().items
        
        usernames = [username['userName'] for username in following_results]
        total_tweets = 0
        
        for username in usernames:
            logger.info(f"Scraping tweets from {username}...")
            tweets_df = scrape_user_tweets(username)
            
            if tweets_df is not None:
                tweets_added = save_to_mongodb(tweets_df)
                total_tweets += tweets_added
        
        return total_tweets
        
    except Exception as e:
        logger.error(f"Error scraping following list: {str(e)}")
        raise

def main():
    """Main function to run the scraping process."""
    try:
        # Example usage
        target_username = "darroverse"  # Replace with your target username
        total_tweets = scrape_following_list(target_username)
        
        logger.info(f"Scraping completed. Total tweets collected: {total_tweets}")
        
    except Exception as e:
        logger.error(f"Error in main scraping process: {str(e)}")
        raise

if __name__ == "__main__":
    main()







