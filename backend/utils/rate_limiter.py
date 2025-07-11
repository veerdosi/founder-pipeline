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
