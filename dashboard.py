import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from database import TwitterDatabase
import json
from datetime import datetime, timedelta
import numpy as np
import subprocess
import time
import os

# Set page config with a more professional theme
st.set_page_config(
    page_title="Tweet Optimizer Dashboard",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        margin-top: 1em;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stExpander {
        background-color: #f0f2f6;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    .stTextArea>div>div>textarea {
        background-color: #f0f2f6;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
db = TwitterDatabase()

def system_status():
    """Display system status and database information"""
    st.header("System Status")
    
    try:
        # Get database stats
        stats = db.get_tweet_stats()
        
        # Display database statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Tweets", f"{stats['total_tweets']:,}")
        with col2:
            st.metric("Average Engagement", f"{stats['avg_engagement']:.2f}")
        with col3:
            st.metric("Top Authors", f"{len(stats['top_authors'])}")
        
        # Display top authors
        st.subheader("Top Authors by Engagement")
        if stats['top_authors']:
            authors_df = pd.DataFrame(stats['top_authors'])
            authors_df.columns = ['Author', 'Tweet Count', 'Avg Engagement']
            st.dataframe(authors_df, use_container_width=True)
        
        # Display top topics
        st.subheader("Top Topics")
        if stats['top_topics']:
            topics_df = pd.DataFrame(stats['top_topics'])
            topics_df.columns = ['Topic', 'Count', 'Avg Engagement']
            st.dataframe(topics_df, use_container_width=True)
        
        # Display database health
        st.subheader("Database Health")
        health_col1, health_col2 = st.columns(2)
        
        with health_col1:
            # Get unprocessed tweets count
            unprocessed_count = db.tweets.count_documents({
                "$or": [
                    {"processed": None},
                    {"processed": ""},
                    {"text_length": None},
                    {"sentiment": None},
                    {"topics": None},
                    {"topics": ""},
                    {"topics": "[]"},
                    {"topics": "null"}
                ]
            })
            st.metric("Unprocessed Tweets", f"{unprocessed_count:,}")
        
        with health_col2:
            # Get error tweets count
            error_count = db.tweets.count_documents({"processed": "error"})
            st.metric("Error Tweets", f"{error_count:,}")
        
        # Display processing status if available
        try:
            with open('processing_progress.json', 'r') as f:
                progress_data = json.load(f)
                st.subheader("Processing Status")
                progress_col1, progress_col2, progress_col3 = st.columns(3)
                
                with progress_col1:
                    st.metric(
                        "Processed Tweets",
                        f"{progress_data['total_processed']:,}"
                    )
                with progress_col2:
                    st.metric(
                        "Processing Speed",
                        f"{progress_data['tweets_per_second']:.2f} tweets/s"
                    )
                with progress_col3:
                    st.metric(
                        "Time Remaining",
                        f"{progress_data['estimated_time_remaining']:.1f}s"
                    )
                
                st.progress(progress_data['progress_percentage'] / 100)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
    except Exception as e:
        st.error(f"Error getting database stats: {str(e)}")
        st.exception(e)

# Title and description with better formatting
st.title("Tweet Optimization Dashboard")
st.markdown("""
    <div style='background-color: #f0f2f6; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
        <p style='margin: 0;'>This dashboard provides a comprehensive interface for optimizing tweets, analyzing trends, and getting actionable recommendations.</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar with better organization
with st.sidebar:
    st.title("Navigation")
    page = st.selectbox(
        "Select Page",
        ["Tweet Optimizer", "Analytics", "History", "Prompt Performance", "Admin"]
    )
    
    st.markdown("---")
    st.markdown("### API Configuration")
    api_key = st.text_input("API Key", type="password")
    
    st.markdown("---")
    st.markdown("### System Status")
    system_status()

# API endpoint
API_URL = "http://localhost:8000"

def get_trending_topics():
    """Get trending topics from recent tweets"""
    try:
        # Get tweets from last 24 hours
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": datetime.datetime.now() - datetime.timedelta(days=1)
                    },
                    "topics": {"$ne": None, "$ne": ""}
                }
            },
            {
                "$group": {
                    "_id": "$topics",
                    "count": {"$sum": 1},
                    "avg_engagement": {"$avg": "$engagement_score"}
                }
            },
            {"$sort": {"avg_engagement": -1}},
            {"$limit": 20}
        ]
        
        results = list(db.tweets.aggregate(pipeline))
        if results:
            df = pd.DataFrame(results)
            df.columns = ['topics', 'count', 'avg_engagement']
            return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting trending topics: {str(e)}")
        return pd.DataFrame()

def get_top_tweets():
    """Get top performing tweets"""
    try:
        pipeline = [
            {
                "$match": {
                    "created_at": {
                        "$gte": datetime.datetime.now() - datetime.timedelta(days=7)
                    }
                }
            },
            {
                "$project": {
                    "text": 1,
                    "author_username": 1,
                    "engagement_score": 1,
                    "created_at": 1,
                    "topics": 1
                }
            },
            {"$sort": {"engagement_score": -1}},
            {"$limit": 10}
        ]
        
        results = list(db.tweets.aggregate(pipeline))
        if results:
            return pd.DataFrame(results)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error getting top tweets: {str(e)}")
        return pd.DataFrame()

def load_performance_metrics():
    """Load performance metrics from file"""
    try:
        with open('performance_metrics.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return None

def load_optimization_history():
    """Load optimization history from file"""
    try:
        with open('optimization_history.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def admin_page():
    """Admin page with processing controls"""
    st.title("Admin Dashboard")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Tweet Processing", "Tweet Collection", "Database Management"])
    
    with tab1:
        st.header("Tweet Processing Configuration")
        
        # Processing parameters in an expander
        with st.expander("Processing Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                batch_size = st.number_input("Batch Size", min_value=1, max_value=100, value=10)
                process_topics = st.checkbox("Process Topics", value=True)
                max_retries = st.number_input("Max Retries", min_value=1, max_value=5, value=3)
                delay_between = st.number_input("Delay Between Batches (seconds)", min_value=0, max_value=10, value=2)
            
            with col2:
                max_tweets = st.number_input("Maximum Tweets to Process", min_value=0, max_value=1000, value=50)
                skip_short_replies = st.checkbox("Skip Short Replies", value=True)
                min_word_count = st.number_input("Minimum Word Count", min_value=1, max_value=10, value=3)
                process_engagement = st.checkbox("Process Engagement Metrics", value=True)
        
        # Process button with status
        if st.button("Start Processing", type="primary"):
            try:
                # Create placeholders for progress and logs
                progress_placeholder = st.empty()
                log_placeholder = st.empty()
                metrics_placeholder = st.empty()
                
                # Save config
                config = {
                    'batch_size': batch_size,
                    'process_topics': process_topics,
                    'max_retries': max_retries,
                    'delay_between': delay_between,
                    'max_tweets': max_tweets,
                    'skip_short_replies': skip_short_replies,
                    'min_word_count': min_word_count,
                    'process_engagement': process_engagement
                }
                with open('processing_config.json', 'w') as f:
                    json.dump(config, f)
                
                # Run the processing script
                log_placeholder.text("Starting tweet processing...")
                
                # Use subprocess.Popen to get real-time output
                process = subprocess.Popen(
                    ['python3', 'process_tweets.py'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Create progress bar
                progress_bar = progress_placeholder.progress(0)
                
                # Read and display output in real-time
                while True:
                    output = process.stdout.readline()
                    if output == '' and process.poll() is not None:
                        break
                    if output:
                        log_placeholder.text(output.strip())
                        
                        # Try to read progress data
                        try:
                            with open('processing_progress.json', 'r') as f:
                                progress_data = json.load(f)
                                
                                # Update progress bar
                                progress_bar.progress(progress_data['progress_percentage'] / 100)
                                
                                # Show metrics in columns
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Processed Tweets",
                                        f"{progress_data['total_processed']:,}"
                                    )
                                with col2:
                                    st.metric(
                                        "Processing Speed",
                                        f"{progress_data['tweets_per_second']:.2f} tweets/s"
                                    )
                                with col3:
                                    st.metric(
                                        "Time Remaining",
                                        f"{progress_data['estimated_time_remaining']:.1f}s"
                                    )
                                
                                # Show detailed metrics
                                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                                with metrics_col1:
                                    st.metric(
                                        "Skipped Tweets",
                                        f"{progress_data['total_skipped']:,}"
                                    )
                                with metrics_col2:
                                    st.metric(
                                        "Error Tweets",
                                        f"{progress_data['total_errors']:,}"
                                    )
                                with metrics_col3:
                                    st.metric(
                                        "Total Handled",
                                        f"{progress_data['total_processed'] + progress_data['total_skipped'] + progress_data['total_errors']:,}"
                                    )
                                
                                # Show status
                                if progress_data.get('status') == 'completed':
                                    st.success("Processing completed successfully!")
                                    break
                                
                        except (FileNotFoundError, json.JSONDecodeError):
                            pass
                
                # Get final output
                stdout, stderr = process.communicate()
                
                if process.returncode == 0:
                    st.success("Processing completed successfully!")
                    st.rerun()
                else:
                    st.error(f"Processing failed with return code {process.returncode}")
                    if stderr:
                        st.error(f"Error output: {stderr}")
            
            except Exception as e:
                st.error(f"Error during processing: {str(e)}")
                st.exception(e)
        
        # Show current processing status if available
        try:
            with open('processing_progress.json', 'r') as f:
                progress_data = json.load(f)
                st.subheader("Current Processing Status")
                
                # Show progress bar
                st.progress(progress_data['progress_percentage'] / 100)
                
                # Show metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Processed Tweets",
                        f"{progress_data['total_processed']:,}"
                    )
                with col2:
                    st.metric(
                        "Processing Speed",
                        f"{progress_data['tweets_per_second']:.2f} tweets/s"
                    )
                with col3:
                    st.metric(
                        "Time Remaining",
                        f"{progress_data['estimated_time_remaining']:.1f}s"
                    )
                
                # Show detailed metrics
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric(
                        "Skipped Tweets",
                        f"{progress_data['total_skipped']:,}"
                    )
                with metrics_col2:
                    st.metric(
                        "Error Tweets",
                        f"{progress_data['total_errors']:,}"
                    )
                with metrics_col3:
                    st.metric(
                        "Total Handled",
                        f"{progress_data['total_processed'] + progress_data['total_skipped'] + progress_data['total_errors']:,}"
                    )
                
                # Show status
                if progress_data.get('status') == 'completed':
                    st.success("Processing completed successfully!")
                elif progress_data.get('status') == 'processing':
                    st.info("Processing in progress...")
        except (FileNotFoundError, json.JSONDecodeError):
            pass
    
    with tab2:
        st.header("Tweet Collection Configuration")
        
        # Collection parameters in an expander
        with st.expander("Collection Parameters", expanded=True):
            col1, col2 = st.columns(2)
            with col1:
                max_usernames = st.number_input("Max Users to Collect", min_value=1, max_value=100, value=50)
                window_length = st.selectbox(
                    "Time Window",
                    ["1d", "7d", "14d", "30d", "90d"],
                    index=2
                )
                max_tweets_per_user = st.number_input("Max Tweets per User", min_value=1, max_value=200, value=50)
            
            with col2:
                min_retweets = st.number_input("Minimum Retweets", min_value=0, max_value=100, value=0)
                min_likes = st.number_input("Minimum Likes", min_value=0, max_value=100, value=0)
                min_replies = st.number_input("Minimum Replies", min_value=0, max_value=100, value=0)
                include_replies = st.checkbox("Include Replies", value=False)
        
        # Collection button with status
        if st.button("Start Collection", type="primary"):
            try:
                st.info("Starting tweet collection...")
                
                # Save collection config
                collection_config = {
                    'max_usernames': max_usernames,
                    'window_length': window_length,
                    'max_tweets_per_user': max_tweets_per_user,
                    'min_retweets': min_retweets,
                    'min_likes': min_likes,
                    'min_replies': min_replies,
                    'include_replies': include_replies
                }
                with open('collection_config.json', 'w') as f:
                    json.dump(collection_config, f)
                
                result = subprocess.run(['python3', 'collect_tweets.py'], capture_output=True, text=True)
                
                if result.returncode == 0:
                    st.success("Tweet collection completed!")
                    st.rerun()
                else:
                    st.error(f"Collection failed: {result.stderr}")
            except Exception as e:
                st.error(f"Error collecting tweets: {str(e)}")
    
    with tab3:
        st.header("Database Management")
        
        # Database maintenance options
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retrain Model", type="secondary"):
                try:
                    st.info("Starting model retraining...")
                    result = subprocess.run(['python3', 'train_model.py'], capture_output=True, text=True)
                    
                    if result.returncode == 0:
                        st.success("Model retraining completed!")
                    else:
                        st.error(f"Retraining failed: {result.stderr}")
                except Exception as e:
                    st.error(f"Error retraining model: {str(e)}")
        
        with col2:
            if st.button("Optimize Database", type="secondary"):
                try:
                    st.info("Starting database optimization...")
                    # Create indexes for better performance
                    db.tweets.create_index("uniqueid", unique=True)
                    db.tweets.create_index("processed")
                    db.tweets.create_index("created_at")
                    db.tweets.create_index("engagement_score")
                    st.success("Database optimization completed!")
                except Exception as e:
                    st.error(f"Error optimizing database: {str(e)}")
        
        # Database information
        st.subheader("Database Information")
        try:
            # Get collection stats
            stats = db.tweets.stats()
            
            # Create a DataFrame for better display
            stats_df = pd.DataFrame([
                {"Metric": "Total Documents", "Value": stats.get("count", 0)},
                {"Metric": "Storage Size", "Value": f"{stats.get('storageSize', 0) / 1024 / 1024:.2f} MB"},
                {"Metric": "Index Size", "Value": f"{stats.get('indexSize', 0) / 1024 / 1024:.2f} MB"},
                {"Metric": "Average Document Size", "Value": f"{stats.get('avgObjSize', 0) / 1024:.2f} KB"}
            ])
            st.dataframe(stats_df, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error getting database info: {str(e)}")
        
        # Database statistics
        st.subheader("Database Statistics")
        try:
            # Get total tweets
            total_tweets = db.tweets.count_documents({})
            
            # Get average engagement
            pipeline = [
                {"$group": {"_id": None, "avg": {"$avg": "$engagement_score"}}}
            ]
            result = list(db.tweets.aggregate(pipeline))
            avg_engagement = result[0]['avg'] if result else 0
            
            # Get unique topics count
            pipeline = [
                {"$group": {"_id": "$topics"}},
                {"$count": "unique_topics"}
            ]
            result = list(db.tweets.aggregate(pipeline))
            unique_topics = result[0]['unique_topics'] if result else 0
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Tweets", f"{total_tweets:,}")
            with col2:
                st.metric("Average Engagement", f"{avg_engagement:.2f}")
            with col3:
                st.metric("Unique Topics", f"{unique_topics:,}")
            
            # Add topic distribution visualization
            st.subheader("Topic Distribution")
            pipeline = [
                {"$group": {
                    "_id": "$topics",
                    "count": {"$sum": 1}
                }},
                {"$sort": {"count": -1}},
                {"$limit": 20}
            ]
            topic_data = list(db.tweets.aggregate(pipeline))
            
            if topic_data:
                topics_df = pd.DataFrame(topic_data)
                topics_df.columns = ['Topic', 'Count']
                fig = px.bar(
                    topics_df,
                    x='Topic',
                    y='Count',
                    title='Top 20 Topics',
                    labels={'Topic': 'Topic', 'Count': 'Number of Tweets'}
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No topic data available yet.")
                
        except Exception as e:
            st.error(f"Error getting database statistics: {str(e)}")
            st.exception(e)

def optimize_tweet_page():
    """Tweet optimization interface"""
    st.title("🎯 Tweet Optimizer")
    
    # Input section
    st.header("Input Tweet")
    tweet_text = st.text_area(
        "Enter your tweet:",
        max_chars=280,
        height=100,
        placeholder="Type your tweet here..."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        n_variations = st.slider("Number of variations", 3, 10, 5)
    with col2:
        n_iterations = st.slider("Number of optimization iterations", 1, 5, 3)
    
    if st.button("Optimize Tweet"):
        if tweet_text:
            with st.spinner("Optimizing your tweet..."):
                # Run optimization
                optimized_tweet, history = optimizer.optimize_tweet(
                    tweet_text,
                    n_iterations=n_iterations,
                    n_variations=n_variations
                )
                
                # Display results
                st.header("Results")
                
                # Original vs Optimized
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Original Tweet")
                    st.text_area("", tweet_text, height=100, disabled=True)
                    st.metric("Predicted Engagement", f"{history[0]['predicted_engagement']:.2f}")
                
                with col2:
                    st.subheader("Optimized Tweet")
                    st.text_area("", optimized_tweet, height=100, disabled=True)
                    st.metric(
                        "Predicted Engagement",
                        f"{history[-1]['predicted_engagement']:.2f}",
                        f"{((history[-1]['predicted_engagement'] / history[0]['predicted_engagement']) - 1) * 100:.1f}%"
                    )
                
                # Optimization Progress
                st.subheader("Optimization Progress")
                progress_df = pd.DataFrame([
                    {
                        'Iteration': entry['iteration'],
                        'Engagement': entry['predicted_engagement']
                    }
                    for entry in history
                ])
                
                fig = px.line(
                    progress_df,
                    x='Iteration',
                    y='Engagement',
                    title='Engagement Score Progress'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show variations from last iteration
                st.subheader("Final Variations")
                variations = history[-1].get('variations', [])
                scores = [env.predict_engagement(var) for var in variations]
                
                variations_df = pd.DataFrame({
                    'Tweet': variations,
                    'Predicted Engagement': scores
                })
                variations_df = variations_df.sort_values('Predicted Engagement', ascending=False)
                
                for i, row in variations_df.iterrows():
                    with st.expander(f"Variation {i+1} (Score: {row['Predicted Engagement']:.2f})"):
                        st.text(row['Tweet'])
        else:
            st.error("Please enter a tweet to optimize")

def analytics_page():
    """Analytics dashboard"""
    st.title("📊 Analytics Dashboard")
    
    # Load data
    performance = load_performance_metrics()
    history = load_optimization_history()
    
    if performance:
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Sample Size", performance['sample_size'])
        with col2:
            st.metric("Mean Absolute Error", f"{performance['mean_absolute_error']:.4f}")
        with col3:
            st.metric("Exploration Error", f"{performance['exploration_error']:.4f}")
        with col4:
            st.metric("Exploitation Error", f"{performance['exploitation_error']:.4f}")
    
    # Engagement distribution
    st.header("Engagement Distribution")
    engagement_stats = db.get_engagement_stats()
    if engagement_stats:
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=[h['predicted_engagement'] for h in history if 'predicted_engagement' in h],
            name='Predicted'
        ))
        fig.add_trace(go.Box(
            y=[h['actual_engagement'] for h in history if 'actual_engagement' in h],
            name='Actual'
        ))
        st.plotly_chart(fig, use_container_width=True)
    
    # Topic distribution
    st.header("Topic Distribution")
    try:
        topics_dict = db.get_topic_distribution()
        if topics_dict:
            # Convert to DataFrame for plotting
            topics_df = pd.DataFrame(list(topics_dict.items()), columns=['Topic', 'Count'])
            topics_df = topics_df.head(10)  # Get top 10 topics
            
            fig = px.bar(topics_df, x='Topic', y='Count', title='Top 10 Topics')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No topic data available yet. Topics will appear here once tweets are processed.")
    except Exception as e:
        st.warning("Unable to load topic distribution. This feature will be available once tweets are processed.")
        st.error(f"Error: {str(e)}")
    
    # Feature importance
    st.header("Feature Importance")
    try:
        importance = db.get_feature_importance()
        if importance:
            importance_df = pd.DataFrame(list(importance.items()), columns=['Feature', 'Importance'])
            importance_df = importance_df.sort_values('Importance', ascending=False)
            fig = px.bar(importance_df, x='Feature', y='Importance', title='Feature Importance')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No feature importance data available yet. This will appear once the model is trained.")
    except Exception as e:
        st.warning("Unable to load feature importance. This feature will be available once the model is trained.")
        st.error(f"Error: {str(e)}")

def history_page():
    """Historical results viewer"""
    st.title("📜 Optimization History")
    
    history = load_optimization_history()
    if history:
        # Convert to dataframe
        df = pd.DataFrame(history)
        df['timestamp'] = pd.to_datetime(df['feedback_collected_at']) if 'feedback_collected_at' in df.columns else pd.Timestamp.now()
        
        # Filters
        col1, col2 = st.columns(2)
        with col1:
            date_range = st.date_input(
                "Date Range",
                value=(
                    df['timestamp'].min().date(),
                    df['timestamp'].max().date()
                )
            )
        with col2:
            min_engagement = st.number_input(
                "Minimum Engagement Score",
                value=0.0,
                step=10.0
            )
        
        # Filter data
        mask = (
            (df['timestamp'].dt.date >= date_range[0]) &
            (df['timestamp'].dt.date <= date_range[1]) &
            (df['predicted_engagement'] >= min_engagement)
        )
        filtered_df = df[mask]
        
        # Display results
        st.header(f"Results ({len(filtered_df)} entries)")
        
        for _, row in filtered_df.iterrows():
            with st.expander(f"Tweet (Score: {row['predicted_engagement']:.2f})"):
                st.text(row['tweet'])
                if 'actual_engagement' in row:
                    st.metric(
                        "Engagement",
                        f"{row['actual_engagement']:.2f}",
                        f"{((row['actual_engagement'] / row['predicted_engagement']) - 1) * 100:.1f}%"
                    )
    else:
        st.info("No optimization history available yet. History will appear here once you optimize some tweets.")

def prompt_performance_page():
    """Prompt performance tracking and analysis"""
    st.title("📈 Prompt Performance Tracking")
    
    # Load prompt history
    try:
        with open('prompt_history.json', 'r') as f:
            prompt_history = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        st.warning("No prompt history available yet.")
        return
    
    if not prompt_history['prompts']:
        st.info("Start optimizing tweets to build prompt performance history.")
        return
    
    # Convert history to DataFrame for easier analysis
    df = pd.DataFrame(prompt_history['performance_metrics'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['prompt'] = prompt_history['prompts']
    
    # Sort by timestamp
    df = df.sort_values('timestamp')
    
    # Key metrics overview
    st.header("Key Performance Metrics")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Average Engagement",
            f"{df['avg_engagement'].mean():.2f}",
            f"{df['avg_engagement'].iloc[-1] - df['avg_engagement'].iloc[0]:.2f}"
        )
    with col2:
        st.metric(
            "Improvement Rate",
            f"{df['improvement_rate'].mean():.1f}%",
            f"{df['improvement_rate'].iloc[-1] - df['improvement_rate'].iloc[0]:.1f}%"
        )
    with col3:
        st.metric(
            "Consistency Score",
            f"{df['consistency'].mean():.2f}",
            f"{df['consistency'].iloc[-1] - df['consistency'].iloc[0]:.2f}"
        )
    
    # Performance over time
    st.header("Performance Over Time")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['avg_engagement'],
        name='Average Engagement',
        line=dict(color='blue')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['max_engagement'],
        name='Max Engagement',
        line=dict(color='green', dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'],
        y=df['min_engagement'],
        name='Min Engagement',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title='Engagement Metrics Over Time',
        xaxis_title='Time',
        yaxis_title='Engagement Score',
        hovermode='x unified'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Content analysis
    st.header("Content Analysis")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(
            df,
            x='timestamp',
            y=['hashtag_ratio', 'emoji_ratio', 'question_ratio', 'exclamation_ratio'],
            title='Content Elements Over Time',
            labels={'value': 'Ratio', 'variable': 'Element Type'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.scatter(
            df,
            x='avg_length',
            y='avg_engagement',
            size='topic_coverage',
            color='consistency',
            title='Tweet Length vs Engagement',
            labels={
                'avg_length': 'Average Tweet Length',
                'avg_engagement': 'Average Engagement',
                'topic_coverage': 'Topic Coverage',
                'consistency': 'Consistency Score'
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Prompt evolution
    st.header("Prompt Evolution")
    for i, (prompt, metrics) in enumerate(zip(df['prompt'], df.to_dict('records'))):
        with st.expander(f"Prompt Version {i+1} ({metrics['timestamp']})"):
            st.text(prompt)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Engagement", f"{metrics['avg_engagement']:.2f}")
            with col2:
                st.metric("Improvement Rate", f"{metrics['improvement_rate']:.1f}%")
            with col3:
                st.metric("Consistency", f"{metrics['consistency']:.2f}")

# Main content with error handling
try:
    if page == "Admin":
        admin_page()
    else:
        # Only import and initialize these components for other pages
        from tweet_rl import TweetEnvironment, TweetOptimizer
        from feedback_collector import FeedbackCollector
        
        env = TweetEnvironment()
        optimizer = TweetOptimizer(env)
        collector = FeedbackCollector(db, env, optimizer)
        
        if page == "Tweet Optimizer":
            optimize_tweet_page()
        elif page == "Analytics":
            analytics_page()
        elif page == "History":
            history_page()
        elif page == "Prompt Performance":
            prompt_performance_page()
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.exception(e)

# Footer with better styling
st.sidebar.markdown("---")
st.sidebar.markdown("""
    <div style='text-align: center;'>
        <p style='margin: 0;'>Made with ❤️ by Your AI Assistant</p>
        <p style='margin: 0; font-size: 0.8em;'>Version 1.0.0</p>
    </div>
""", unsafe_allow_html=True) 