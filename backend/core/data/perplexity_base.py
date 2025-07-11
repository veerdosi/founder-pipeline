"""Base class for Perplexity AI services providing common functionality."""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
from abc import ABC, abstractmethod

from ..config import settings
from ...utils.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


class PerplexityBaseService(ABC):
    """Base service class for Perplexity AI API interactions."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'perplexity_api_key', None)
        self.base_url = "https://api.perplexity.ai"
        self.rate_limiter = RateLimiter(max_requests=10, time_window=60)  # 10 requests per minute for Perplexity
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Common configuration for all Perplexity requests
        self.default_model = "sonar-pro"
        self.default_temperature = 0.1
        self.default_max_tokens = 1500
        self.default_timeout = 60
        
        # High-quality domains for search filtering (max 10 for Perplexity API)
        self.trusted_domains = [
            "techcrunch.com", "forbes.com", "bloomberg.com", "crunchbase.com",
            "linkedin.com", "reuters.com", "wsj.com", "ft.com", "cnbc.com",
            "sec.gov"
        ]
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def query_perplexity(
        self, 
        query: str, 
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        search_recency: str = "month"
    ) -> Optional[Dict[str, Any]]:
        """
        Make a query to Perplexity AI with comprehensive error handling.
        
        Args:
            query: The search query
            system_prompt: Custom system prompt (optional)
            model: Model to use (defaults to sonar-pro)
            temperature: Temperature for generation (defaults to 0.1)
            max_tokens: Maximum tokens to generate (defaults to 1500)
            search_recency: Recency filter for search results
            
        Returns:
            Dictionary containing the response or None if failed
        """
        if not self.api_key:
            logger.error("❌ Perplexity API key not configured")
            return None
            
        await self.rate_limiter.acquire()
        
        logger.debug(f"🔍 Making Perplexity query: {query[:100]}...")
        
        if not self.session:
            self.session = aiohttp.ClientSession()
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        messages = [
            {
                "role": "system",
                "content": system_prompt or self._get_default_system_prompt()
            },
            {
                "role": "user",
                "content": query
            }
        ]
        
        payload = {
            "model": model or self.default_model,
            "messages": messages,
            "max_tokens": max_tokens or self.default_max_tokens,
            "temperature": temperature or self.default_temperature,
            "top_p": 0.9,
            "search_domain_filter": self.trusted_domains,
            "return_citations": True,
            "search_recency_filter": search_recency,
            "top_k": 0,
            "stream": False,
            "presence_penalty": 0,
            "frequency_penalty": 1
        }
        
        # Retry logic for failed API calls
        max_retries = 2
        for attempt in range(max_retries + 1):
            try:
                async with self.session.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=self.default_timeout
                ) as response:
                    logger.debug(f"📡 Perplexity API response status: {response.status}")
                    
                    if response.status == 200:
                        try:
                            response_data = await response.json()
                            logger.debug(f"✅ Perplexity API response received successfully")
                            return response_data
                        except json.JSONDecodeError as e:
                            logger.error(f"❌ Failed to parse Perplexity JSON response: {e}")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ Perplexity API error {response.status}: {error_text}")
                        return None
                        
            except asyncio.TimeoutError:
                if attempt < max_retries:
                    logger.warning(f"⚠️ Perplexity API timeout (attempt {attempt + 1}/{max_retries + 1}), retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ Perplexity API timeout after {self.default_timeout} seconds (all retries exhausted)")
                    return None
            except aiohttp.ClientError as e:
                if attempt < max_retries:
                    logger.warning(f"⚠️ HTTP client error (attempt {attempt + 1}/{max_retries + 1}): {e}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                    continue
                else:
                    logger.error(f"❌ HTTP client error calling Perplexity API: {e}")
                    return None
            except Exception as e:
                logger.error(f"❌ Unexpected error calling Perplexity API: {e}", exc_info=True)
                return None
    
    def extract_content_from_response(self, response_data: Dict[str, Any]) -> Optional[str]:
        """Extract the main content from Perplexity response."""
        try:
            if not isinstance(response_data, dict):
                logger.error(f"Response data is not a dict: {type(response_data)}")
                return None
            
            if 'choices' not in response_data or not response_data['choices']:
                logger.warning("No choices in Perplexity response")
                return None
            
            choice = response_data['choices'][0]
            message = choice.get('message', {})
            content = message.get('content', '')
            
            if not isinstance(content, str):
                content = str(content)
            
            return content
            
        except Exception as e:
            logger.error(f"Error extracting content from Perplexity response: {e}")
            return None
    
    def extract_citations_from_response(self, response_data: Dict[str, Any]) -> List[str]:
        """Extract citations from Perplexity response."""
        citations = []
        
        try:
            if isinstance(response_data, dict) and 'citations' in response_data:
                if isinstance(response_data['citations'], list):
                    citations = [
                        citation.get('url', '') 
                        for citation in response_data['citations']
                        if isinstance(citation, dict) and citation.get('url')
                    ]
        except Exception as e:
            logger.warning(f"Error extracting citations: {e}")
        
        return citations
    
    def parse_structured_data(self, content: str, data_type: str) -> List[Dict[str, Any]]:
        """
        Parse structured data from Perplexity content based on data type.
        
        Args:
            content: The content to parse
            data_type: Type of data to extract (financial, media, etc.)
            
        Returns:
            List of structured data dictionaries
        """
        try:
            # Truncate content to prevent processing issues
            content = content[:3000]
            
            # Split content into sentences for processing
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
            
            structured_data = []
            
            for sentence in sentences[:20]:  # Limit to first 20 sentences
                if self._is_relevant_sentence(sentence, data_type):
                    data_item = self._extract_structured_item(sentence, data_type)
                    if data_item:
                        structured_data.append(data_item)
            
            return structured_data
            
        except Exception as e:
            logger.error(f"Error parsing structured data: {e}")
            return []
    
    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt for Perplexity queries."""
        return """You are a comprehensive business intelligence assistant specializing in founder and company research. 
        Provide accurate, detailed, and well-sourced information. Always include specific dates, numbers, and financial details when available.
        Be thorough but concise, and prioritize recent and verifiable information from authoritative sources.
        Structure your response with clear facts and avoid speculation."""
    
    def _is_relevant_sentence(self, sentence: str, data_type: str) -> bool:
        """Check if a sentence is relevant for the specified data type."""
        sentence_lower = sentence.lower()
        
        if data_type == 'financial':
            return any(keyword in sentence_lower for keyword in [
                'founded', 'co-founded', 'acquired', 'sold', 'ipo', 'exit', 'investment',
                'valuation', 'funding', 'raised', 'million', 'billion', 'board', 'angel'
            ])
        elif data_type == 'media':
            return any(keyword in sentence_lower for keyword in [
                'awarded', 'received', 'won', 'recognition', 'honor', 'interview',
                'featured', 'published', 'speaker', 'keynote', 'conference', 'ted'
            ])
        elif data_type == 'biographical':
            return any(keyword in sentence_lower for keyword in [
                'graduated', 'degree', 'university', 'college', 'education',
                'worked', 'served', 'experience', 'career', 'background'
            ])
        
        return False
    
    def _extract_structured_item(self, sentence: str, data_type: str) -> Optional[Dict[str, Any]]:
        """Extract a structured data item from a sentence."""
        try:
            # Basic extraction - can be enhanced with more sophisticated parsing
            return {
                'type': data_type,
                'content': sentence,
                'extracted_at': datetime.now().isoformat(),
                'confidence': 0.8 if len(sentence) > 50 else 0.6
            }
        except Exception as e:
            logger.warning(f"Error extracting structured item: {e}")
            return None
    
    @abstractmethod
    async def collect_data(self, founder_name: str, current_company: str) -> Any:
        """Abstract method to be implemented by subclasses."""
        pass
    
    @abstractmethod
    def get_query_templates(self) -> Dict[str, List[str]]:
        """Abstract method to get query templates specific to the service."""
        pass