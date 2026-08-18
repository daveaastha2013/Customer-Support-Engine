import json
import hashlib
import logging
from typing import Optional, Dict, Any
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class RedisCacheManager:
    """Optional Redis caching manager with graceful degradation."""
    
    def __init__(self, host: str = None, port: int = None, ttl: int = None):
        self.host = host or settings.REDIS_HOST
        self.port = port or settings.REDIS_PORT
        self.ttl = ttl or settings.REDIS_TTL_SECONDS
        self.client = None
        self.enabled = False
        
        try:
            import redis
            self.client = redis.Redis(host=self.host, port=self.port, socket_timeout=1.0)
            # Test ping
            self.client.ping()
            self.enabled = True
            logger.info(f"Connected to Redis server at {self.host}:{self.port}")
        except Exception as e:
            logger.info(f"Redis is unavailable ({e}). Continuing with caching DISABLED (graceful fallback).")
            self.enabled = False

    def _get_key(self, query: str) -> str:
        hash_str = hashlib.sha256(query.strip().lower().encode('utf-8')).hexdigest()
        return f"rag:query_cache:{hash_str}"

    def get(self, query: str) -> Optional[Dict[str, Any]]:
        if not self.enabled or not self.client:
            return None
            
        try:
            key = self._get_key(query)
            data = self.client.get(key)
            if data:
                logger.info(f"Cache HIT for query: '{query[:30]}...'")
                res = json.loads(data)
                res["cached"] = True
                return res
        except Exception as e:
            logger.warning(f"Error reading from Redis cache: {e}")
            
        return None

    def set(self, query: str, data: Dict[str, Any]):
        if not self.enabled or not self.client:
            return
            
        try:
            key = self._get_key(query)
            # Do not cache error responses or non-serializable objects
            clean_data = dict(data)
            clean_data["cached"] = True
            self.client.setex(key, self.ttl, json.dumps(clean_data))
            logger.info(f"Cached query response in Redis with TTL={self.ttl}s")
        except Exception as e:
            logger.warning(f"Error writing to Redis cache: {e}")
