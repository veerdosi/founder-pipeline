"""Simple but effective rate limiting utility for API requests."""

import asyncio
import time
from typing import List
import logging

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple rate limiter for API requests with sliding window."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_requests: Maximum number of requests allowed in the time window
            time_window: Time window in seconds (default: 60 seconds)
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests: List[float] = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """
        Acquire permission to make a request.
        
        This method will block if the rate limit would be exceeded,
        waiting until a request slot becomes available.
        """
        async with self._lock:
            now = time.time()
            
            # Remove requests that are outside the time window
            self.requests = [req_time for req_time in self.requests 
                            if now - req_time < self.time_window]
            
            # Check if we're at the limit
            if len(self.requests) >= self.max_requests:
                # Calculate how long to wait
                oldest_request = min(self.requests)
                wait_time = self.time_window - (now - oldest_request)
                
                if wait_time > 0:
                    logger.debug(f"Rate limit reached, waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)
                    # Recursively call to check again after waiting
                    return await self.acquire()
            
            # Add the current request timestamp
            self.requests.append(now)
    
    def can_make_request(self) -> bool:
        """
        Check if a request can be made without waiting.
        
        Returns:
            True if a request can be made immediately, False otherwise
        """
        now = time.time()
        
        # Remove requests that are outside the time window
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        return len(self.requests) < self.max_requests
    
    def get_current_usage(self) -> dict:
        """
        Get current rate limiter statistics.
        
        Returns:
            Dictionary with current usage statistics
        """
        now = time.time()
        
        # Clean up old requests
        self.requests = [req_time for req_time in self.requests 
                        if now - req_time < self.time_window]
        
        return {
            'current_requests': len(self.requests),
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'utilization_percentage': (len(self.requests) / self.max_requests) * 100,
            'requests_remaining': self.max_requests - len(self.requests)
        }
    
    def reset(self) -> None:
        """Reset the rate limiter by clearing all request history."""
        self.requests.clear()


class TokenBucketRateLimiter:
    """Token bucket rate limiter for more sophisticated rate limiting."""
    
    def __init__(self, max_tokens: int, refill_rate: float):
        """
        Initialize token bucket rate limiter.
        
        Args:
            max_tokens: Maximum number of tokens in the bucket
            refill_rate: Number of tokens added per second
        """
        self.max_tokens = max_tokens
        self.refill_rate = refill_rate
        self.tokens = max_tokens
        self.last_refill = time.time()
        self._lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> None:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire (default: 1)
        """
        async with self._lock:
            await self._refill_tokens()
            
            while self.tokens < tokens:
                # Calculate how long to wait for enough tokens
                wait_time = (tokens - self.tokens) / self.refill_rate
                await asyncio.sleep(wait_time)
                await self._refill_tokens()
            
            self.tokens -= tokens
    
    async def _refill_tokens(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.refill_rate
        self.tokens = min(self.max_tokens, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def get_current_tokens(self) -> int:
        """Get the current number of available tokens."""
        return int(self.tokens)


# Global rate limiters for common use cases
default_rate_limiter = RateLimiter(max_requests=60, time_window=60)  # 60 requests per minute
openai_rate_limiter = RateLimiter(max_requests=50, time_window=60)   # 50 requests per minute for OpenAI
perplexity_rate_limiter = RateLimiter(max_requests=20, time_window=60)  # 20 requests per minute for Perplexity