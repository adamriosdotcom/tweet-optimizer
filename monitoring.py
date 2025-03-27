"""
Monitoring and metrics collection for the tweet optimization system.
"""

from prometheus_client import Counter, Histogram, Gauge
from prometheus_fastapi_instrumentator import Instrumentator
from fastapi import FastAPI
import time

# Define metrics
PREDICT_REQUESTS = Counter(
    'tweet_predict_requests_total',
    'Total number of tweet prediction requests',
    ['status']
)

OPTIMIZE_REQUESTS = Counter(
    'tweet_optimize_requests_total',
    'Total number of tweet optimization requests',
    ['status']
)

PREDICT_LATENCY = Histogram(
    'tweet_predict_latency_seconds',
    'Latency of tweet prediction requests',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)

OPTIMIZE_LATENCY = Histogram(
    'tweet_optimize_latency_seconds',
    'Latency of tweet optimization requests',
    buckets=[0.5, 1.0, 2.0, 5.0, 10.0]
)

ACTIVE_REQUESTS = Gauge(
    'tweet_active_requests',
    'Number of currently active requests',
    ['endpoint']
)

PREDICTED_ENGAGEMENT = Histogram(
    'tweet_predicted_engagement',
    'Distribution of predicted engagement scores',
    buckets=[10, 25, 50, 100, 250, 500]
)

OPTIMIZATION_IMPROVEMENT = Histogram(
    'tweet_optimization_improvement',
    'Distribution of optimization improvements (%)',
    buckets=[10, 25, 50, 100, 200, 500]
)

def init_monitoring(app: FastAPI):
    """Initialize monitoring for the FastAPI application"""
    Instrumentator().instrument(app).expose(app)

class RequestMetrics:
    """Context manager for request metrics"""
    def __init__(self, endpoint: str, metric_type: str):
        self.endpoint = endpoint
        self.metric_type = metric_type
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        ACTIVE_REQUESTS.labels(endpoint=self.endpoint).inc()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        ACTIVE_REQUESTS.labels(endpoint=self.endpoint).dec()
        
        if self.metric_type == 'predict':
            PREDICT_LATENCY.observe(duration)
            PREDICT_REQUESTS.labels(status='success' if not exc_type else 'error').inc()
        elif self.metric_type == 'optimize':
            OPTIMIZE_LATENCY.observe(duration)
            OPTIMIZE_REQUESTS.labels(status='success' if not exc_type else 'error').inc()

def record_prediction(engagement: float):
    """Record prediction metrics"""
    PREDICTED_ENGAGEMENT.observe(engagement)

def record_optimization(improvement: float):
    """Record optimization metrics"""
    OPTIMIZATION_IMPROVEMENT.observe(improvement) 