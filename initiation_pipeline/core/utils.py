"""Core utilities and common functionality."""

import asyncio
import logging
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TypeVar, Union

import aiofiles
import pandas as pd
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from tenacity import retry, stop_after_attempt, wait_exponential

from .config import settings

T = TypeVar('T')

# Global console for rich output
console = Console()


def setup_logging() -> None:
    """Set up logging with Rich handler."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True)
        ]
    )
    
    # File logging if specified
    if settings.log_file:
        file_handler = logging.FileHandler(settings.log_file)
        file_handler.setFormatter(
            logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        )
        logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class RateLimiter:
    """Simple async rate limiter."""
    
    def __init__(self, max_requests: int, time_window: int = 60):
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = []
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> None:
        """Acquire a rate limit slot."""
        async with self._lock:
            now = time.time()
            # Remove old requests
            self.requests = [
                req_time for req_time in self.requests 
                if now - req_time < self.time_window
            ]
            
            if len(self.requests) >= self.max_requests:
                # Wait until we can make a request
                sleep_time = self.time_window - (now - self.requests[0]) + 0.1
                await asyncio.sleep(sleep_time)
                return await self.acquire()
            
            self.requests.append(now)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10)
)
async def http_request_with_retry(
    session,
    method: str,
    url: str,
    **kwargs
) -> Any:
    """Make HTTP request with retry logic."""
    async with getattr(session, method.lower())(url, **kwargs) as response:
        response.raise_for_status()
        return await response.json()


async def save_checkpoint(data: Any, filename: str) -> None:
    """Save checkpoint data asynchronously."""
    if not settings.checkpoint_enabled:
        return
        
    checkpoint_path = settings.default_output_dir / f"checkpoint_{filename}.pkl"
    
    try:
        async with aiofiles.open(checkpoint_path, 'wb') as f:
            await f.write(pickle.dumps(data))
        console.print(f"✅ Checkpoint saved: {checkpoint_path}")
    except Exception as e:
        console.print(f"❌ Error saving checkpoint: {e}")


async def load_checkpoint(filename: str) -> Optional[Any]:
    """Load checkpoint data asynchronously."""
    checkpoint_path = settings.default_output_dir / f"checkpoint_{filename}.pkl"
    
    if not checkpoint_path.exists():
        return None
        
    try:
        async with aiofiles.open(checkpoint_path, 'rb') as f:
            data = await f.read()
        console.print(f"✅ Checkpoint loaded: {checkpoint_path}")
        return pickle.loads(data)
    except Exception as e:
        console.print(f"❌ Error loading checkpoint: {e}")
        return None


def save_to_csv(data: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """Save data to CSV file."""
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    console.print(f"✅ Data saved to CSV: {output_path}")


async def save_to_json(data: Any, output_path: Union[str, Path]) -> None:
    """Save data to JSON file asynchronously."""
    import json
    
    async with aiofiles.open(output_path, 'w') as f:
        await f.write(json.dumps(data, indent=2, default=str))
    console.print(f"✅ Data saved to JSON: {output_path}")


def validate_api_keys() -> bool:
    """Validate that required API keys are present."""
    missing_keys = []
    
    if not settings.exa_api_key or settings.exa_api_key == "your_exa_api_key_here":
        missing_keys.append("EXA_API_KEY")
    if not settings.openai_api_key or settings.openai_api_key == "your_openai_api_key_here":
        missing_keys.append("OPENAI_API_KEY")
    if not settings.serper_api_key or settings.serper_api_key == "your_serper_api_key_here":
        missing_keys.append("SERPER_API_KEY")
    if not settings.apify_api_key or settings.apify_api_key == "your_apify_api_key_here":
        missing_keys.append("APIFY_API_KEY")
    if not settings.perplexity_api_key or settings.perplexity_api_key == "your_perplexity_api_key_here":
        missing_keys.append("PERPLEXITY_API_KEY")
    if not settings.crunchbase_api_key or settings.crunchbase_api_key == "your_crunchbase_api_key_here":
        missing_keys.append("CRUNCHBASE_API_KEY")
    
    if missing_keys:
        console.print("❌ Missing required API keys:")
        for key in missing_keys:
            console.print(f"   - {key}")
        console.print("\nPlease set these in your .env file or environment variables.")
        return False
    
    console.print("✅ All required API keys are configured")
    return True


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = " ".join(text.split())
    
    # Remove any <think> tags from LLM output
    import re
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    return text.strip()


def extract_year_from_date(date_str: str) -> Optional[int]:
    """Extract year from various date string formats."""
    if not date_str:
        return None
        
    try:
        return pd.to_datetime(date_str).year
    except:
        # Try to extract year with regex
        import re
        year_match = re.search(r'\b(19|20)\d{2}\b', str(date_str))
        if year_match:
            return int(year_match.group())
    
    return None


def create_progress_bar() -> Progress:
    """Create a rich progress bar."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    )


class Timer:
    """Simple context manager for timing operations."""
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        console.print(f"⏱️  {self.description}: {elapsed:.2f}s")


def deduplicate_by_key(items: List[Dict[str, Any]], key: str) -> List[Dict[str, Any]]:
    """Deduplicate a list of dictionaries by a specific key."""
    seen = set()
    unique_items = []
    
    for item in items:
        value = item.get(key)
        if value and value not in seen:
            seen.add(value)
            unique_items.append(item)
    
    return unique_items


def chunk_list(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into chunks of specified size."""
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


async def run_with_semaphore(
    semaphore: asyncio.Semaphore,
    coro_func,
    *args,
    **kwargs
):
    """Run a coroutine with semaphore-based concurrency control."""
    async with semaphore:
        return await coro_func(*args, **kwargs)
