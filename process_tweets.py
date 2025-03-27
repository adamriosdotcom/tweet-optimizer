#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced tweet feature engineering pipeline with error handling.
"""

import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import datetime
import emoji
import textstat
import time
import json
import logging
from openai import OpenAI
from dotenv import load_dotenv
import os
from database import TwitterDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("tweet_processor")

# Load environment variables
load_dotenv()

# Initialize database and OpenAI client
db = TwitterDatabase()
openai_client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# System prompt for topic extraction
system_prompt = """
Extract 1-5 concise topics from this tweet. Topics should be conceptual entities, actions, or sentiments.
For short replies or reactions, consider the sentiment or action being expressed (e.g. 'agreement', 'appreciation', 'humor').
Don't include generic categories like 'announcement', 'question', 'thought'.
Return ONLY a JSON object with format {"topics": ["topic1", "topic2"]}.
If no meaningful topics can be identified, return {"topics": []}.
"""

def extract_text_features(text):
    """Extract features from tweet text"""
    if not isinstance(text, str):
        return {
            'text_length': 0,
            'word_count': 0,
            'hashtag_count': 0,
            'mention_count': 0,
            'emoji_count': 0,
            'flesch_reading_ease': 0,
            'sentiment': 0,
            'subjectivity': 0
        }
    
    # Basic text features
    text_length = len(text)
    words = re.findall(r'\b\w+\b', text.lower())
    word_count = len(words)
    
    # Count hashtags and mentions
    hashtag_count = len(re.findall(r'#\w+', text))
    mention_count = len(re.findall(r'@\w+', text))
    
    # Count emojis
    emoji_count = sum(1 for char in text if char in emoji.EMOJI_DATA)
    
    # Readability
    flesch_reading_ease = textstat.flesch_reading_ease(text) if word_count > 0 else 0
    
    # Sentiment analysis
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    return {
        'text_length': text_length,
        'word_count': word_count,
        'hashtag_count': hashtag_count,
        'mention_count': mention_count,
        'emoji_count': emoji_count,
        'flesch_reading_ease': flesch_reading_ease,
        'sentiment': sentiment,
        'subjectivity': subjectivity
    }

def extract_engagement_features(row):
    """Extract engagement-based features"""
    # Convert to numeric with fallback to 0
    like_count = pd.to_numeric(row.get('likeCount', 0), errors='coerce') or 0
    retweet_count = pd.to_numeric(row.get('retweetCount', 0), errors='coerce') or 0
    reply_count = pd.to_numeric(row.get('replyCount', 0), errors='coerce') or 0
    quote_count = pd.to_numeric(row.get('quoteCount', 0), errors='coerce') or 0
    
    # Engagement score (weighted sum)
    engagement_score = (
        1.0 * like_count + 
        2.0 * retweet_count + 
        1.5 * reply_count + 
        1.5 * quote_count
    )
    
    # Followers-based normalization
    followers = pd.to_numeric(row.get('author_followers', 0), errors='coerce') or 1
    popularity_index = engagement_score / np.sqrt(followers)
    
    return {
        'engagement_score': engagement_score,
        'popularity_index': popularity_index,
    }

def extract_author_features(row):
    """Extract author-related features"""
    try:
        # Account age
        created_at = pd.to_datetime(row.get('author_createdAt', None))
        if created_at is not None:
            current_time = pd.to_datetime('now')
            account_age_days = (current_time - created_at).days
        else:
            account_age_days = 0
            
        # Follower/following ratio
        followers = pd.to_numeric(row.get('author_followers', 0), errors='coerce') or 0
        following = pd.to_numeric(row.get('author_following', 0), errors='coerce') or 1
        follower_following_ratio = followers / following
            
        return {
            'account_age_days': account_age_days,
            'follower_following_ratio': follower_following_ratio,
        }
    except Exception as e:
        logger.error(f"Error extracting author features: {e}")
        return {
            'account_age_days': 0,
            'follower_following_ratio': 0,
        }

def extract_topics_with_llm(text, client, retries=3, delay=2):
    """Extract topics using OpenAI with retry logic"""
    if not isinstance(text, str):
        logger.warning(f"Invalid text type: {type(text)}")
        return {"topics": []}
        
    text = text.strip()
    words = text.split()
    
    # Skip very short replies that are just mentions
    if text.startswith('@') and len(words) <= 3:
        logger.info(f"Skipping short reply tweet: {text}")
        return {"topics": ["short reply"]}
        
    for attempt in range(retries):
        try:
            logger.info(f"Attempting to extract topics from text: {text[:100]}...")
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "system", "content": system_prompt}, 
                          {"role": "user", "content": text}],
                response_format={"type": "json_object"}
            )
            result = json.loads(response.choices[0].message.content)
            logger.info(f"Extracted topics: {result.get('topics', [])}")
            return result
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Error parsing API response: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            logger.error(f"API error: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    logger.error("Failed to extract topics after all retries")
    return {"topics": []}  # Fallback response

def process_tweet_batch(batch_size=10, process_topics=True):
    """Process a batch of tweets from the database"""
    # Get tweets that need processing
    query = """
    SELECT * FROM tweets 
    WHERE (
        -- Check for truly unprocessed tweets
        processed IS NULL
        OR processed = ''
        -- Include tweets missing required fields that haven't been processed
        OR (
            (text_length IS NULL OR sentiment IS NULL)
            AND processed IS NULL
        )
        -- Include tweets that have been processed but have empty topics
        OR (
            topics IS NULL 
            OR topics = ''
            OR topics = '[]'
            OR topics = 'null'
        )
    )
    AND processed != 'skipped'  -- Don't process skipped tweets
    AND processed != 'completed'  -- Don't reprocess completed tweets
    AND processed != 'error'  -- Don't reprocess error tweets
    LIMIT ?
    """
    
    conn = db.conn
    df = pd.read_sql_query(query, conn, params=(batch_size,))
    
    if len(df) == 0:
        logger.info("No tweets need processing")
        return 0
    
    logger.info(f"Processing {len(df)} tweets")
    
    # Log the first few tweets we're processing
    for i, row in df.head(3).iterrows():
        logger.info(f"Sample tweet to process: {row.get('text', '')[:100]}...")
    
    # Process each tweet
    features = []
    for i, row in df.iterrows():
        try:
            logger.info(f"Processing tweet {i+1}/{len(df)}")
            
            # Check if tweet should be skipped
            text = row.get('text', '')
            words = text.split()
            
            # Skip very short replies that are just mentions
            if text.startswith('@') and len(words) <= 3:
                logger.info(f"Skipping short reply tweet: {text}")
                # Mark skipped tweets as processed with minimal features
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE tweets 
                    SET text_length = ?,
                        word_count = ?,
                        sentiment = 0,
                        topics = 'short reply',
                        processed = 'skipped'
                    WHERE uniqueid = ?
                """, (len(text), len(words), row['uniqueid']))
                conn.commit()
                continue
            
            # Extract text features
            text_features = extract_text_features(text)
            logger.debug(f"Text features extracted: {text_features}")
            
            # Extract engagement features
            engagement_features = extract_engagement_features(row)
            logger.debug(f"Engagement features extracted: {engagement_features}")
            
            # Extract author features
            author_features = extract_author_features(row)
            logger.debug(f"Author features extracted: {author_features}")
            
            # Topic extraction (only if requested)
            if process_topics:
                topics_data = extract_topics_with_llm(text, openai_client)
                topics_list = topics_data.get('topics', [])
                topics_str = ','.join(topics_list)
                logger.debug(f"Topics extracted: {topics_str}")
            else:
                topics_str = row.get('topics', '')
            
            # Combine all features
            tweet_features = {
                'uniqueid': row['uniqueid'],
                'topics': topics_str,
                **text_features,
                **engagement_features,
                **author_features,
                'processed': 'completed'  # Mark as successfully processed
            }
            
            features.append(tweet_features)
            logger.info(f"Successfully processed tweet {i+1}")
            
        except Exception as e:
            logger.error(f"Error processing tweet {i+1}: {str(e)}")
            # Mark as error but don't stop processing
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE tweets SET processed = 'error' WHERE uniqueid = ?",
                (row['uniqueid'],)
            )
            conn.commit()
            continue
    
    if features:
        # Update database with all features
        features_df = pd.DataFrame(features)
        feature_columns = [col for col in features_df.columns if col != 'uniqueid']
        
        try:
            # Update features and ensure transaction is committed
            updated_count = db.update_features(features_df, feature_columns)
            logger.info(f"Database update returned count: {updated_count}")
            
            # Ensure transaction is committed
            db.conn.commit()
            
            # Verify the update with a more comprehensive query
            verify_query = """
            SELECT COUNT(*) as count 
            FROM tweets 
            WHERE uniqueid IN ({})
            AND (
                (processed = 'completed' AND text_length IS NOT NULL AND sentiment IS NOT NULL AND topics IS NOT NULL AND topics != '')
                OR (processed = 'skipped')
            )
            """.format(','.join(['?'] * len(features_df)))
            
            verified_count = pd.read_sql_query(
                verify_query,
                db.conn,
                params=features_df['uniqueid'].tolist()
            )['count'].iloc[0]
            
            logger.info(f"Verification query found {verified_count} updated tweets")
            
            # Log the uniqueids being verified
            logger.info(f"Verifying tweets with IDs: {features_df['uniqueid'].tolist()}")
            
            return verified_count
            
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            db.conn.rollback()  # Rollback on error
            return 0
    
    return 0

def main():
    """Main processing function"""
    logger.info("Starting tweet processing")
    
    # Set configuration for 50 tweets
    batch_size = 10
    process_topics = True
    max_retries = 3
    delay_between = 2
    max_tweets = 50  # Process only 50 tweets
    
    # Get total number of unprocessed tweets
    query = """
    SELECT COUNT(*) as count FROM tweets 
    WHERE text_length IS NULL 
    OR sentiment IS NULL 
    OR topics IS NULL 
    OR topics = ''
    """
    total_unprocessed = pd.read_sql_query(query, db.conn)['count'].iloc[0]
    
    if max_tweets:
        total_unprocessed = min(total_unprocessed, max_tweets)
    
    logger.info(f"Found {total_unprocessed} tweets to process")
    
    # Process tweets in smaller batches
    total_processed = 0
    start_time = time.time()
    
    while True:
        # Check if we've hit the max tweets limit
        if max_tweets and total_processed >= max_tweets:
            logger.info(f"Reached maximum tweets limit of {max_tweets}")
            break
            
        # Calculate remaining tweets for this batch
        remaining = min(batch_size, total_unprocessed - total_processed)
        if remaining <= 0:
            break
            
        processed_count = process_tweet_batch(remaining, process_topics)
        total_processed += processed_count
        
        # Calculate processing speed and estimated time remaining
        elapsed_time = time.time() - start_time
        tweets_per_second = float(total_processed / elapsed_time if elapsed_time > 0 else 0)
        remaining_tweets = total_unprocessed - total_processed
        estimated_time = float(remaining_tweets / tweets_per_second if tweets_per_second > 0 else 0)
        
        # Log progress
        progress = float((total_processed / total_unprocessed) * 100)
        logger.info(f"Progress: {progress:.1f}% ({total_processed}/{total_unprocessed} tweets processed)")
        logger.info(f"Processing speed: {tweets_per_second:.2f} tweets/second")
        logger.info(f"Estimated time remaining: {estimated_time:.1f} seconds")
        
        # Save progress to file for dashboard
        progress_data = {
            'total_processed': int(total_processed),
            'total_unprocessed': int(total_unprocessed),
            'progress_percentage': float(progress),
            'tweets_per_second': float(tweets_per_second),
            'estimated_time_remaining': float(estimated_time),
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        with open('processing_progress.json', 'w') as f:
            json.dump(progress_data, f)
        
        if processed_count < remaining:
            logger.warning(f"Processed fewer tweets than expected: {processed_count}/{remaining}")
            break
        
        time.sleep(delay_between)  # Use configured delay between batches
    
    logger.info(f"Processing complete. Total tweets processed: {total_processed}")
    
    # Save final progress
    final_progress = {
        'total_processed': int(total_processed),
        'total_unprocessed': int(total_unprocessed),
        'progress_percentage': 100,
        'tweets_per_second': float(total_processed / (time.time() - start_time) if (time.time() - start_time) > 0 else 0),
        'estimated_time_remaining': 0,
        'timestamp': datetime.datetime.now().isoformat()
    }
    
    with open('processing_progress.json', 'w') as f:
        json.dump(final_progress, f)

if __name__ == "__main__":
    main() 