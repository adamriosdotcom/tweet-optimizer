"""
Caching module for the tweet optimization system.
"""

import redis
import json
import logging
from typing import Optional, Any
from config import settings

logger = logging.getLogger(__name__)

class Cache:
    """Redis-based caching system"""
    
    def __init__(self):
        """Initialize Redis connection"""
        try:
            self.redis = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=settings.REDIS_DB,
                password=settings.REDIS_PASSWORD,
                decode_responses=True
            )
            logger.info("Redis cache initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Redis cache: {str(e)}")
            self.redis = None
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self.redis:
            return None
            
        try:
            value = self.redis.get(key)
            return json.loads(value) if value else None
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {str(e)}")
            return None
    
    def set(self, key: str, value: Any, expire: int = 3600) -> bool:
        """Set value in cache with expiration"""
        if not self.redis:
            return False
            
        try:
            return self.redis.setex(
                key,
                expire,
                json.dumps(value)
            )
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {str(e)}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        if not self.redis:
            return False
            
        try:
            return bool(self.redis.delete(key))
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {str(e)}")
            return False
    
    def clear_pattern(self, pattern: str) -> bool:
        """Clear all keys matching pattern"""
        if not self.redis:
            return False
            
        try:
            keys = self.redis.keys(pattern)
            if keys:
                return bool(self.redis.delete(*keys))
            return True
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {str(e)}")
            return False

# Global cache instance
cache = Cache() 