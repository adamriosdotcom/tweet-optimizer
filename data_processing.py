import sqlite3
import pandas as pd
import numpy as np
import re
from textblob import TextBlob
import datetime
import emoji
import textstat
from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel
import time
import json

# Load data from SQLite
conn = sqlite3.connect('tweets.db')
df = pd.read_sql_query("SELECT * FROM tweets", conn)
conn.close()

# Add at the beginning of your database setup
def ensure_database_indexes(db_path='tweets.db'):
    """Ensure necessary indexes exist for query performance"""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Add indexes for commonly queried columns
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_author_username ON tweets(author.userName)",
        "CREATE INDEX IF NOT EXISTS idx_created_at ON tweets(createdAt)",
        "CREATE INDEX IF NOT EXISTS idx_like_count ON tweets(likeCount)",
        "CREATE INDEX IF NOT EXISTS idx_topics ON tweets(topics)"
    ]
    
    for index_query in indexes:
        try:
            cursor.execute(index_query)
        except sqlite3.OperationalError as e:
            print(f"Index creation error (may already exist): {e}")
    
    conn.commit()
    conn.close()

# Database indexing (one-time operation)
import sqlite3
import pandas as pd

def update_db_batch(df_batch, db_path='tweets.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    new_feature_columns = [
        'text_length', 'word_count', 'hashtag_count', 'mention_count', 'sentiment',
        'engagement_score', 'rolling_engagement', 'account_age_days',
        'follower_following_ratio', 'emoji_count', 'flesch_reading_ease',
        'subjectivity', 'daily_author_activity', 'interaction_reciprocity',
        'popularity_index', 'topics'
    ]

    set_clause = ', '.join([f"{col} = ?" for col in new_feature_columns])
    sql = f"UPDATE tweets SET {set_clause} WHERE uniqueid = ?"

    update_data = [
        tuple(row[col] for col in new_feature_columns) + (row['uniqueid'],)
        for _, row in df_batch.iterrows()
    ]

    cursor.executemany(sql, update_data)
    conn.commit()
    conn.close()

    print(f"Updated {len(df_batch)} rows successfully.")




# Connect to your database
conn = sqlite3.connect('tweets.db')
cursor = conn.cursor()

# Add new columns if they don't already exist
new_columns = [
    ('text_length', 'INTEGER'),
    ('word_count', 'INTEGER'),
    ('hashtag_count', 'INTEGER'),
    ('mention_count', 'INTEGER'),
    ('sentiment', 'REAL'),
    ('engagement_score', 'REAL'),
    ('rolling_engagement', 'REAL'),
    ('account_age_days', 'INTEGER'),
    ('follower_following_ratio', 'REAL'),
    ('emoji_count', 'INTEGER'),
    ('flesch_reading_ease', 'REAL'),
    ('subjectivity', 'REAL'),
    ('daily_author_activity', 'REAL'),
    ('interaction_reciprocity', 'INTEGER'),
    ('popularity_index', 'REAL'),
    ('topics', 'TEXT')
]

for column_name, column_type in new_columns:
    try:
        cursor.execute(f"ALTER TABLE tweets ADD COLUMN {column_name} {column_type};")
        print(f"Added column {column_name}.")
    except sqlite3.OperationalError as e:
        print(f"Column {column_name} already exists.")

conn.commit()
conn.close()


# Ensure createdAt is datetime
df['createdAt'] = pd.to_datetime(df['createdAt'], errors='coerce')

numeric_cols = ['likeCount', 'retweetCount', 'replyCount', 'quoteCount', 'bookmarkCount']

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df['engagement_score'] = (
    df['likeCount'] * 1 +
    df['retweetCount'] * 2 +
    df['replyCount'] * 1.5 +
    df['quoteCount'] * 2.5 +
    df['bookmarkCount'] * 1
)
# Conditional feature computation helper
def conditional_compute(df, column_name, func):
    if column_name not in df or df[column_name].isna().any():
        df[column_name] = df.apply(func, axis=1)

# Create features conditionally
conditional_compute(df, 'text_length', lambda row: len(str(row['text'])))
conditional_compute(df, 'word_count', lambda row: len(str(row['text']).split()))
conditional_compute(df, 'hashtag_count', lambda row: len(re.findall(r"#\w+", str(row['text']))))
conditional_compute(df, 'mention_count', lambda row: len(re.findall(r"@\w+", str(row['text']))))
conditional_compute(df, 'sentiment', lambda row: TextBlob(str(row['text'])).sentiment.polarity)
conditional_compute(df, 'emoji_count', lambda row: len(emoji.emoji_list(str(row['text']))))
conditional_compute(df, 'flesch_reading_ease', lambda row: textstat.flesch_reading_ease(str(row['text'])))
conditional_compute(df, 'subjectivity', lambda row: TextBlob(str(row['text'])).sentiment.subjectivity)

# Engagement Score
df['engagement_score'] = (
    df['likeCount'] * 1 + df['retweetCount'] * 2 + df['replyCount'] * 1.5 +
    df['quoteCount'] * 2.5 + df['bookmarkCount'] * 1
)

# Rolling Engagement
df.sort_values(['author.id', 'createdAt'], inplace=True)
df['rolling_engagement'] = df.groupby('author.id')['engagement_score'].transform(lambda x: x.rolling(5, min_periods=1).mean())

# Account age in days
conditional_compute(
    df,
    'account_age_days',
    lambda row: (
        datetime.datetime.now(datetime.timezone.utc) - 
        pd.to_datetime(row['author.createdAt'], errors='coerce').tz_convert('UTC')
    ).days
)
# Follower/Following ratio
conditional_compute(df, 'follower_following_ratio', lambda row: pd.to_numeric(row['author.followers'], errors='coerce') / (pd.to_numeric(row['author.following'], errors='coerce') + 1))

# Daily author activity
conditional_compute(df, 'daily_author_activity', lambda row: pd.to_numeric(row['author.statusesCount'], errors='coerce') / (row['account_age_days'] + 1))

# Interaction reciprocity
conditional_compute(df, 'interaction_reciprocity', lambda row: int(bool(row.get('isReply')) or bool(row.get('isQuote'))))

# Popularity index
conditional_compute(df, 'popularity_index', lambda row: row['engagement_score'] / (pd.to_numeric(row['author.followers'], errors='coerce') + 1))

# --- GPT Topic Generation (conditionally computed) ---
import openai
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv
import pandas as pd
import time

load_dotenv()
client = OpenAI()

class TweetTopics(BaseModel):
    tweet_id: str
    topics: list[str]

batch_size = 100
rows_processed = 0
total_rows = len(df)

for idx, row in df.iterrows():
    current_topics = row.get('topics', '')
    
    # Skip already computed topics (including placeholder spaces)
    if pd.notna(current_topics) and current_topics.strip() != "":
        continue

    tweet = row['text']

    completion = client.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that extracts only interesting conceptual topics from tweets like vibe coding, OpenAI, San Francisco, and avoid things like technology, congratulations, and community."
                    " Generate a maximum of 10 standardized topics suitable for grouping similar tweets together."
                    " Avoid overly specific details or common filler words."
                    " If unclear, return an empty list."
                    " Output JSON schema: {'tweet_id': <string>, 'topics': [<strings>]}."
                )
            },
            {
                "role": "user",
                "content": tweet
            }
        ],
        response_format=TweetTopics,
    )

    tweet_topics = completion.choices[0].message.parsed
    topics_list = tweet_topics.model_dump()['topics']

    # Set placeholder if topics empty
    df.at[idx, 'topics'] = " " if not topics_list else ",".join(topics_list).lower()

    rows_processed += 1
    time.sleep(0.1)

    # Update database every 500 tweets
    if rows_processed % batch_size == 0:
        batch_df = df.iloc[idx - batch_size + 1: idx + 1]
        update_db_batch(batch_df)
        print("100 tweets processed. Database updated.")

# Final batch update for remaining rows
remaining_rows = rows_processed % batch_size
if remaining_rows:
    batch_df = df.iloc[-remaining_rows:]
    update_db_batch(batch_df)

print("All topics generated and database updated.")




## TRENDING TOPICS ALIGNMENT




# # --- Step 1: Clean and split topics ---
# df['topics_list'] = df['topics'].apply(
#     lambda x: [topic.strip() for topic in re.split(r',\s*', x) if topic.strip()]
# )

# # --- Step 2: Compute document frequency ---
# num_docs = len(df)
# doc_freq = Counter()
# for topics in df['topics_list']:
#     unique_topics = set(topics)
#     doc_freq.update(unique_topics)

# # --- Step 3: Generate generic topics dynamically ---
# threshold = 0.2
# generic_topics = {topic for topic, count in doc_freq.items() if count / num_docs >= threshold}
# print("Generic topics (dynamically generated):", generic_topics)

# # --- Step 4 (MODIFIED): Count unique authors per topic ---
# # Explode dataframe to have one topic per row, paired with author.id
# exploded_authors_topics = df.explode('topics_list')[['author.id', 'topics_list']].drop_duplicates()

# # Now count unique authors per topic
# unique_authors_per_topic = exploded_authors_topics.groupby('topics_list')['author.id'].nunique().reset_index()
# unique_authors_per_topic.columns = ['topic', 'unique_author_count']

# # Define popular topics as the top 30 topics by unique authors, excluding generic
# popular_topics = set(
#     unique_authors_per_topic[~unique_authors_per_topic['topic'].str.lower().isin(
#         {g.lower() for g in generic_topics}
#     )].sort_values(by='unique_author_count', ascending=False).head(30)['topic']
# )

# print("Popular topics (by unique authors, excluding generic):", popular_topics)

# # --- Step 5: Define similarity metric functions ---
# def intersection_count(tweet_topics, popular_set):
#     return len(set(tweet_topics).intersection(popular_set))

# def jaccard_similarity(tweet_topics, popular_set):
#     tweet_set = set(tweet_topics)
#     union = tweet_set.union(popular_set)
#     return len(tweet_set.intersection(popular_set)) / len(union) if union else 0

# # --- Step 6: Apply similarity metrics ---
# df['common_topic_count'] = df['topics_list'].apply(lambda x: intersection_count(x, popular_topics))
# df['jaccard_similarity'] = df['topics_list'].apply(lambda x: jaccard_similarity(x, popular_topics))


# df = df






## TRENDING TOPICS BY RANGE



# import sqlite3
# import pandas as pd
# from collections import Counter
# import re
# from datetime import timedelta

# conn = sqlite3.connect('tweets.db')

# # Load recent data (for example, last 30 days)
# df = pd.read_sql("SELECT * FROM tweets WHERE createdAt >= date('now', '-30 days')", conn)

# df['createdAt'] = pd.to_datetime(df['createdAt']).dt.date
# df['topics_list'] = df['topics'].apply(lambda x: [t.strip() for t in re.split(r',\s*', str(x)) if t.strip()])

# # Calculate daily trending topics
# daily_trends = {}

# for day in df['createdAt'].unique():
#     daily_df = df[df['createdAt'] == day]
#     exploded = daily_df.explode('topics_list')
#     exploded = exploded.drop_duplicates(['author.id', 'topics_list'])
    
#     daily_counts = exploded['topics_list'].value_counts()
#     daily_trends[day] = daily_counts.head(30).index.tolist()

# # Calculate topic shifts between days
# topic_shifts = {}

# sorted_days = sorted(daily_trends.keys())

# for i in range(1, len(sorted_days)):
#     previous_day_topics = daily_trends[sorted_days[i-1]]
#     current_day_topics = daily_trends[sorted_days[i]]

#     shifts = {}
#     for topic in current_day_topics:
#         previous_rank = previous_day_topics.index(topic) if topic in previous_day_topics else 31
#         current_rank = current_day_topics.index(topic)
#         shifts[topic] = previous_rank - current_rank  # Positive means trending up

#     topic_shifts[sorted_days[i]] = shifts

# # Example to check shifts:
# for day, shifts in list(topic_shifts.items())[-5:]:
#     print(f"Date: {day}, Topic shifts: {shifts}")

# conn.close()



import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect('tweets.db')
cursor = conn.cursor()

# Define columns you want to update
feature_columns = [
    'text_length', 'word_count', 'hashtag_count', 'mention_count', 'sentiment',
    'engagement_score', 'rolling_engagement', 'account_age_days',
    'follower_following_ratio', 'emoji_count', 'flesch_reading_ease',
    'subjectivity', 'daily_author_activity', 'interaction_reciprocity',
    'popularity_index', 'topics'
]

# Prepare SQL update query
set_clause = ', '.join([f"{col} = ?" for col in feature_columns])
sql = f"UPDATE tweets SET {set_clause} WHERE uniqueid = ?"

# Prepare the data for update
update_data = [
    tuple(row[col] for col in feature_columns) + (row['uniqueid'],)
    for _, row in df.iterrows()
]

# Execute batch update
cursor.executemany(sql, update_data)

# Commit changes and close connection
conn.commit()
conn.close()

print("Database updated successfully.")




# Consolidate the database update functions into a single utility function
def update_db(df, columns, batch_size=500, db_path='tweets.db'):
    """
    Update the database with computed features in efficient batches
    
    Args:
        df: DataFrame with computed features
        columns: List of column names to update
        batch_size: Number of records to update at once
        db_path: Path to SQLite database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Prepare SQL update query
    set_clause = ', '.join([f"{col} = ?" for col in columns])
    sql = f"UPDATE tweets SET {set_clause} WHERE uniqueid = ?"
    
    # Process in batches
    total_rows = len(df)
    for i in range(0, total_rows, batch_size):
        batch_df = df.iloc[i:min(i+batch_size, total_rows)]
        
        # Prepare the data for update
        update_data = [
            tuple(row[col] for col in columns) + (row['uniqueid'],)
            for _, row in batch_df.iterrows()
        ]
        
        # Execute batch update
        cursor.executemany(sql, update_data)
        conn.commit()
        print(f"Updated rows {i} to {min(i+batch_size, total_rows)} of {total_rows}")
    
    conn.close()
    print("Database update completed successfully.")




# Add comprehensive error handling for API calls
def extract_topics_with_llm(text, client, retries=3, delay=2):
    """Extract topics using OpenAI with retry logic"""
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini", 
                messages=[{"role": "system", "content": system_prompt}, 
                          {"role": "user", "content": text}],
                response_format={"type": "json_object"}
            )
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing API response: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
        except Exception as e:
            print(f"API error: {e}")
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))  # Exponential backoff
    
    return {"topics": []}  # Fallback response



