"""Utility functions for data processing and validation."""

import re
from typing import Any, Dict, List, Optional, TypeVar
from urllib.parse import urlparse

import pandas as pd

T = TypeVar('T')


def clean_text(text: str) -> str:
    """Clean and normalize text."""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = " ".join(text.split())
    
    # Remove any <think> tags from LLM output
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text.strip()


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert value to float."""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (ValueError, TypeError):
        return default


def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert value to int."""
    try:
        if value is None or value == "":
            return default
        return int(float(value))  # Handle strings like "123.0"
    except (ValueError, TypeError):
        return default
