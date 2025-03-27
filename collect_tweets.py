#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tweet collection script with better error handling and database integration.
"""

from apify_client import ApifyClient
import pandas as pd
import numpy as np
import sqlite3
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

# Initialize parameters
maxUsernames = 50  # Number of users to collect from
windowLength = "14d"  # Last 14 days
maxTweets = 50  # Tweets per user

# Initialize the Apify client
client = ApifyClient(APIFY_TOKEN)

# Get target usernames
get_users = {
    "getFollowers": False,
    "getFollowing": True,
    "maxItems": maxUsernames,
    "twitterHandles": ["darroverse"]  # We'll get 50 users from darroverse's following list
}

following_actor_id = "apidojo/twitter-user-scraper"
following_run = client.actor(following_actor_id).call(run_input=get_users)

# Get the dataset ID for the results
following_dataset_id = following_run["defaultDatasetId"]

# Fetch the results from the dataset
following_results = client.dataset(following_dataset_id).list_items().items

usernames = [username['userName'] for username in following_results]

rows_added = 0

for username in usernames:
    logger.info(f"Scraping tweets from {username}...")
    
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
        "within_time": windowLength,
        "maxItems": maxTweets,
        "min_retweets": 0,
        "min_faves": 0,
        "min_replies": 0,
        "-min_retweets": 0,
        "-min_faves": 0,
        "-min_replies": 0
    }
    
    try:
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
                
                # Create or update database schema
                columns = tweets.columns
                col_defs = ['uniqueid TEXT PRIMARY KEY']
                for col in columns:
                    if col == "uniqueid":
                        continue
                    col_defs.append(f'"{col}" TEXT')
                
                create_table_query = "CREATE TABLE IF NOT EXISTS tweets (\n" + ",\n".join(col_defs) + "\n);"
                
                conn = sqlite3.connect('tweets.db')
                cursor = conn.cursor()
                cursor.execute(create_table_query)
                conn.commit()
                
                # Add new columns if needed
                cursor.execute("PRAGMA table_info(tweets);")
                existing_columns = [info[1] for info in cursor.fetchall()]
                
                for col in tweets.columns:
                    if col not in existing_columns:
                        alter_query = f'ALTER TABLE tweets ADD COLUMN "{col}" TEXT;'
                        print(f"Adding column: {col}")
                        cursor.execute(alter_query)
                        conn.commit()
                
                # Insert data
                tweets.to_sql('tweets', conn, if_exists='append', index=False)
                rows_added += len(tweets)
                logger.info(f"Added {len(tweets)} tweets for {username}")
                
                conn.close()
            else:
                logger.warning(f"No valid posts to concatenate for {username}")
        else:
            logger.warning(f"No valid tweets found for {username}")
            
    except Exception as e:
        logger.error(f"Error processing tweets for {username}: {str(e)}")
        continue

# Log results
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
log_entry = f"{current_time}: {rows_added} rows added.\n"

with open("scrape_log.txt", "a") as log_file:
    log_file.write(log_entry)

logger.info(f"Collection complete. Total tweets collected: {rows_added}") 