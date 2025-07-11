"""Utility functions for data processing and validation."""

import re
import json
from typing import Any, Dict, List, Optional, TypeVar
from urllib.parse import urlparse
import logging

import pandas as pd

T = TypeVar('T')

logger = logging.getLogger(__name__)


def extract_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Extracts and parses JSON from text using Python's built-in json.loads().
    This is more forgiving than LangChain's overly strict parser.
    """
    if not text or not isinstance(text, str):
        return {}
    
    try:
        # Try to find JSON block first
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Find first complete JSON object
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index == -1 or end_index == -1:
                return {}
            json_str = text[start_index : end_index + 1]
        
        # Try parsing directly first - Python's json.loads is forgiving
        result = json.loads(json_str)
        return result if isinstance(result, dict) else {}
        
    except json.JSONDecodeError as e:
        logger.warning(f"Initial JSON parsing failed: {e}")
        # Try with basic cleaning
        return _clean_and_parse_json(text)
        
    except Exception as e:
        logger.warning(f"Unexpected error during JSON parsing: {e}")
        return {}


def _clean_and_parse_json(text: str) -> Dict[str, Any]:
    """
    Fallback JSON parsing with minimal cleaning.
    """
    try:
        # Find JSON block
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            start_index = text.find('{')
            end_index = text.rfind('}')
            if start_index == -1 or end_index == -1:
                return {}
            json_str = text[start_index : end_index + 1]
        
        # Minimal cleaning - only remove control characters
        json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
        json_str = json_str.strip()
        
        result = json.loads(json_str)
        return result if isinstance(result, dict) else {}
        
    except Exception as e:
        logger.debug(f"Fallback JSON parsing failed: {e}")
        return {}


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