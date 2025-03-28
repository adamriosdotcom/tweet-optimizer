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
    query = {
        "$or": [
            # Check for truly unprocessed tweets (NULL or empty)
            {"processed": None},
            {"processed": ""},
            # Include tweets missing required fields
            {"text_length": None},
            {"sentiment": None},
            # Include tweets that have been processed but have empty topics
            {"topics": None},
            {"topics": ""},
            {"topics": "[]"},
            {"topics": "null"}
        ],
        # Exclude tweets that are already marked as completed, skipped, or error
        "processed": {"$nin": ["completed", "skipped", "error"]}
    }
    
    # Get tweets from MongoDB with the specified batch size
    tweets = list(db.tweets.find(query).limit(batch_size))
    df = pd.DataFrame(tweets) if tweets else pd.DataFrame()
    
    if len(df) == 0:
        # Log the current state of the database
        status = {
            'total': db.tweets.count_documents({}),
            'unprocessed': db.tweets.count_documents({"$or": [{"processed": None}, {"processed": ""}]}),
            'missing_features': db.tweets.count_documents({"$or": [{"text_length": None}, {"sentiment": None}]}),
            'missing_topics': db.tweets.count_documents({"$or": [{"topics": None}, {"topics": ""}, {"topics": "[]"}, {"topics": "null"}]}),
            'completed': db.tweets.count_documents({"processed": "completed"}),
            'skipped': db.tweets.count_documents({"processed": "skipped"}),
            'error': db.tweets.count_documents({"processed": "error"})
        }
        logger.info(f"Database status: {status}")
        logger.info("No tweets need processing")
        return 0
    
    logger.info(f"Processing {len(df)} tweets")
    
    # Process each tweet
    features = []
    processed_count = 0
    skipped_count = 0
    error_count = 0
    
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
                db.tweets.update_one(
                    {"uniqueid": row['uniqueid']},
                    {"$set": {
                        "text_length": len(text),
                        "word_count": len(words),
                        "sentiment": 0,
                        "topics": "short reply",
                        "processed": "skipped"
                    }}
                )
                skipped_count += 1
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
            processed_count += 1
            logger.info(f"Successfully processed tweet {i+1}")
            
        except Exception as e:
            logger.error(f"Error processing tweet {i+1}: {str(e)}")
            # Mark as error but don't stop processing
            db.tweets.update_one(
                {"uniqueid": row['uniqueid']},
                {"$set": {"processed": "error"}}
            )
            error_count += 1
            continue
    
    if features:
        # Update database with all features
        features_df = pd.DataFrame(features)
        feature_columns = [col for col in features_df.columns if col != 'uniqueid']
        
        try:
            # Update features
            updated_count = db.update_features(features_df, feature_columns)
            logger.info(f"Database update returned count: {updated_count}")
            
            # Verify the update
            verify_query = {
                "uniqueid": {"$in": features_df['uniqueid'].tolist()},
                "$or": [
                    {
                        "processed": "completed",
                        "text_length": {"$ne": None},
                        "sentiment": {"$ne": None},
                        "topics": {"$ne": None},
                        "topics": {"$ne": ""}
                    },
                    {"processed": "skipped"}
                ]
            }
            
            verified_count = db.tweets.count_documents(verify_query)
            logger.info(f"Verification query found {verified_count} updated tweets")
            
            return processed_count, skipped_count, error_count
            
        except Exception as e:
            logger.error(f"Failed to update database: {str(e)}")
            return 0, 0, 0
    
    return 0, 0, 0

def main():
    """Main processing function"""
    logger.info("Starting tweet processing")
    
    # Load configuration
    try:
        with open('processing_config.json', 'r') as f:
            config = json.load(f)
            batch_size = config.get('batch_size', 50)
            process_topics = config.get('process_topics', True)
            max_retries = config.get('max_retries', 3)
            delay_between = config.get('delay_between', 1)
            max_tweets = config.get('max_tweets', 5000)  # Default to 5000 if not specified
    except (FileNotFoundError, json.JSONDecodeError):
        # Use default values if config file doesn't exist
        batch_size = 50
        process_topics = True
        max_retries = 3
        delay_between = 1
        max_tweets = 5000
    
    # Get total number of unprocessed tweets
    status = {
        'total': db.tweets.count_documents({}),
        'unprocessed': db.tweets.count_documents({"$or": [{"processed": None}, {"processed": ""}]}),
        'missing_features': db.tweets.count_documents({"$or": [{"text_length": None}, {"sentiment": None}]}),
        'missing_topics': db.tweets.count_documents({"$or": [{"topics": None}, {"topics": ""}, {"topics": "[]"}, {"topics": "null"}]}),
        'completed': db.tweets.count_documents({"processed": "completed"}),
        'skipped': db.tweets.count_documents({"processed": "skipped"}),
        'error': db.tweets.count_documents({"processed": "error"})
    }
    logger.info(f"Initial database status: {status}")
    
    # Get number of tweets that need processing
    process_query = {
        "$or": [
            {"processed": None},
            {"processed": ""},
            {"text_length": None},
            {"sentiment": None},
            {"topics": None},
            {"topics": ""},
            {"topics": "[]"},
            {"topics": "null"}
        ],
        "processed": {"$nin": ["completed", "skipped", "error"]}
    }
    total_unprocessed = db.tweets.count_documents(process_query)
    
    if max_tweets:
        total_unprocessed = min(total_unprocessed, max_tweets)
        logger.info(f"Processing limit set to {max_tweets} tweets")
    
    logger.info(f"Found {total_unprocessed} tweets to process")
    
    # Process tweets in batches
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    start_time = time.time()
    consecutive_empty_batches = 0
    
    while True:
        # Check if we've hit the max tweets limit
        if max_tweets and (total_processed + total_skipped + total_errors) >= max_tweets:
            logger.info(f"Reached maximum tweets limit of {max_tweets}")
            break
            
        # Calculate remaining tweets for this batch
        remaining = min(batch_size, total_unprocessed - (total_processed + total_skipped + total_errors))
        if remaining <= 0:
            logger.info("No more tweets to process")
            break
            
        processed_count, skipped_count, error_count = process_tweet_batch(remaining, process_topics)
        
        # Update counts
        total_processed += processed_count
        total_skipped += skipped_count
        total_errors += error_count
        
        # Track empty batches
        if processed_count == 0 and skipped_count == 0 and error_count == 0:
            consecutive_empty_batches += 1
            if consecutive_empty_batches >= 3:  # Stop after 3 empty batches
                logger.warning("Stopping after 3 consecutive empty batches")
                break
        else:
            consecutive_empty_batches = 0
        
        # Calculate processing speed and estimated time remaining
        elapsed_time = time.time() - start_time
        total_handled = total_processed + total_skipped + total_errors
        tweets_per_second = float(total_handled / elapsed_time if elapsed_time > 0 else 0)
        remaining_tweets = total_unprocessed - total_handled
        estimated_time = float(remaining_tweets / tweets_per_second if tweets_per_second > 0 else 0)
        
        # Log progress
        progress = float((total_handled / total_unprocessed) * 100)
        logger.info(f"Progress: {progress:.1f}% ({total_handled}/{total_unprocessed} tweets handled)")
        logger.info(f"Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
        logger.info(f"Processing speed: {tweets_per_second:.2f} tweets/second")
        logger.info(f"Estimated time remaining: {estimated_time:.1f} seconds")
        
        # Save progress to file for dashboard
        progress_data = {
            'total_processed': int(total_processed),
            'total_skipped': int(total_skipped),
            'total_errors': int(total_errors),
            'total_unprocessed': int(total_unprocessed),
            'progress_percentage': float(progress),
            'tweets_per_second': float(tweets_per_second),
            'estimated_time_remaining': float(estimated_time),
            'timestamp': datetime.datetime.now().isoformat(),
            'status': 'processing'  # Add status field
        }
        
        with open('processing_progress.json', 'w') as f:
            json.dump(progress_data, f)
        
        time.sleep(delay_between)  # Use configured delay between batches
    
    logger.info(f"Processing complete. Total tweets handled: {total_processed + total_skipped + total_errors}")
    logger.info(f"Processed: {total_processed}, Skipped: {total_skipped}, Errors: {total_errors}")
    
    # Save final progress
    final_progress = {
        'total_processed': int(total_processed),
        'total_skipped': int(total_skipped),
        'total_errors': int(total_errors),
        'total_unprocessed': int(total_unprocessed),
        'progress_percentage': 100,
        'tweets_per_second': float((total_processed + total_skipped + total_errors) / (time.time() - start_time) if (time.time() - start_time) > 0 else 0),
        'estimated_time_remaining': 0,
        'timestamp': datetime.datetime.now().isoformat(),
        'status': 'completed'  # Add status field
    }
    
    with open('processing_progress.json', 'w') as f:
        json.dump(final_progress, f)

if __name__ == "__main__":
    main() 