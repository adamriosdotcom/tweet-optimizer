import logging
from feedback_collector import FeedbackCollector
from tweet_rl import TweetEnvironment, TweetOptimizer
from database import TwitterDatabase

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feedback_test.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("feedback_test")

def test_feedback_collection():
    """Test the feedback collection system with historical data"""
    try:
        # Initialize components
        db = TwitterDatabase()
        env = TweetEnvironment()
        optimizer = TweetOptimizer(env)
        collector = FeedbackCollector(db, env, optimizer)
        
        # Test tweet optimization
        test_tweet = "Just released our new AI model! It's faster and more accurate than previous versions. Check it out at our website."
        
        # Optimize the tweet
        optimized_tweet, history = optimizer.optimize_tweet(test_tweet)
        
        # Add to optimization history
        collector.optimization_history.extend(history)
        collector.save_optimization_history()
        
        # Collect feedback
        feedback_count = collector.collect_feedback()
        logger.info(f"Collected feedback for {feedback_count} tweets")
        
        # Analyze performance
        performance = collector.analyze_performance()
        
        if performance:
            logger.info("\nPerformance Analysis Results:")
            logger.info(f"Sample Size: {performance['sample_size']}")
            logger.info(f"Mean Absolute Error: {performance['mean_absolute_error']:.4f}")
            logger.info(f"Mean Relative Error: {performance['mean_relative_error']:.4f}")
            logger.info(f"Exploration Error: {performance['exploration_error']:.4f}")
            logger.info(f"Exploitation Error: {performance['exploitation_error']:.4f}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error in feedback collection test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_feedback_collection()
    if success:
        logger.info("Feedback collection test completed successfully")
    else:
        logger.error("Feedback collection test failed") 