"""Real-time verification service for updating stale founder data using Perplexity."""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import os
import re

from .level_thresholds import DataPoint, SourceType

logger = logging.getLogger(__name__)


@dataclass
class VerificationSource:
    """Configuration for a verification source."""
    name: str
    source_type: SourceType
    api_endpoint: str
    api_key: Optional[str]
    rate_limit_per_minute: int
    timeout_seconds: int = 30


class RealTimeFounderVerifier:
    """Real-time verification for potentially stale founder data using Perplexity."""
    
    def __init__(self):
        self.perplexity_api_key = os.getenv("PERPLEXITY_API_KEY")
        self.session = None
        
        if not self.perplexity_api_key:
            logger.warning("Perplexity API key not configured - real-time verification disabled")
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def verify_founder_data_realtime(
        self, 
        founder_name: str, 
        company_name: str,
        last_update: Optional[datetime] = None,
        verification_queries: Optional[List[str]] = None
    ) -> List[DataPoint]:
        """
        Real-time verification of founder data when significant time has passed since collection.
        
        This is used when:
        - Data was collected weeks/months ago
        - Before final ranking to ensure current accuracy
        - To verify major claims (exits, funding, roles)
        """
        
        logger.info(f"Real-time verification for {founder_name} at {company_name}")
        
        # Check if verification is needed
        if last_update and (datetime.now() - last_update).days < 7:
            logger.info(f"Data is recent ({last_update}), skipping real-time verification")
            return []
        
        verification_data = []
        
        # Use Perplexity for real-time verification
        if self.perplexity_api_key:
            perplexity_data = await self._verify_with_perplexity(
                founder_name, 
                company_name, 
                verification_queries
            )
            verification_data.extend(perplexity_data)
        
        logger.info(f"Real-time verification complete: {len(verification_data)} data points")
        return verification_data
    
    async def _verify_with_perplexity(
        self, 
        founder_name: str, 
        company_name: str,
        custom_queries: Optional[List[str]] = None
    ) -> List[DataPoint]:
        """Verify founder data using Perplexity real-time search."""
        
        if not self.perplexity_api_key:
            return []
        
        try:
            data_points = []
            
            # Default verification queries focused on key claims
            if custom_queries:
                verification_queries = custom_queries
            else:
                verification_queries = [
                    f"{founder_name} {company_name} funding recent news 2024 2025",
                    f"{founder_name} {company_name} IPO acquisition exit recent",
                    f"{founder_name} CEO founder {company_name} current role 2024",
                    f"{founder_name} {company_name} valuation unicorn billion recent"
                ]
            
            headers = {
                "Authorization": f"Bearer {self.perplexity_api_key}",
                "Content-Type": "application/json"
            }
            
            for query in verification_queries:
                try:
                    payload = {
                        "model": "sonar-small-online",  # Real-time model
                        "messages": [
                            {
                                "role": "user",
                                "content": f"""Search for recent factual information about: {query}

Please provide:
1. Current funding status and amounts
2. Recent exits, IPOs, or acquisitions  
3. Current role and company status
4. Any major announcements in 2024-2025

Focus only on verified facts with dates. Return 'No recent information found' if nothing current is available."""
                            }
                        ],
                        "max_tokens": 800,
                        "temperature": 0.1
                    }
                    
                    async with self.session.post(
                        "https://api.perplexity.ai/chat/completions", 
                        headers=headers, 
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=30)
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                            
                            if content and "No recent information found" not in content:
                                # Extract structured data from response
                                verification_data = self._extract_verification_data(content, query)
                                
                                data_points.append(DataPoint(
                                    value=verification_data,
                                    source_type=SourceType.MEDIA_REPORT,
                                    source_url="https://perplexity.ai",
                                    verified=True,
                                    confidence=0.8  # High confidence for recent Perplexity data
                                ))
                                
                                logger.info(f"Perplexity verification found data for: {query}")
                            else:
                                logger.info(f"No recent information found for: {query}")
                        else:
                            logger.warning(f"Perplexity API error {response.status} for query: {query}")
                    
                    # Rate limiting
                    await asyncio.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Perplexity query failed for '{query}': {e}")
                    continue
            
            return data_points
            
        except Exception as e:
            logger.error(f"Perplexity verification failed: {e}")
            return []
    
    def _extract_verification_data(self, content: str, original_query: str) -> Dict[str, Any]:
        """Extract structured data from Perplexity response."""
        
        verification_data = {
            "query": original_query,
            "search_timestamp": datetime.now().isoformat(),
            "raw_content": content,
            "extracted_facts": []
        }
        
        # Extract funding information
        funding_patterns = [
            r'\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)',
            r'raised\s+\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)',
            r'funding\s+of\s+\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)'
        ]
        
        for pattern in funding_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1))
                unit = match.group(2).lower()
                if unit in ['billion', 'b']:
                    amount *= 1000  # Convert to millions
                
                verification_data["extracted_facts"].append({
                    "type": "funding",
                    "amount_millions": amount,
                    "source_text": match.group(0)
                })
        
        # Extract valuation information
        valuation_patterns = [
            r'valued\s+at\s+\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)',
            r'valuation\s+of\s+\$(\d+(?:\.\d+)?)\s*(million|billion|M|B)',
            r'unicorn.*\$(\d+(?:\.\d+)?)\s*(billion|B)'
        ]
        
        for pattern in valuation_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                amount = float(match.group(1))
                unit = match.group(2).lower()
                if unit in ['billion', 'b']:
                    amount *= 1000
                
                verification_data["extracted_facts"].append({
                    "type": "valuation",
                    "amount_millions": amount,
                    "source_text": match.group(0)
                })
        
        # Extract exit/IPO information
        if re.search(r'\b(IPO|ipo|went public|public offering)\b', content, re.IGNORECASE):
            verification_data["extracted_facts"].append({
                "type": "exit",
                "exit_type": "IPO",
                "source_text": "IPO mentioned in content"
            })
        
        if re.search(r'\b(acquired|acquisition|bought by)\b', content, re.IGNORECASE):
            verification_data["extracted_facts"].append({
                "type": "exit", 
                "exit_type": "Acquisition",
                "source_text": "Acquisition mentioned in content"
            })
        
        # Extract dates
        date_patterns = [
            r'\b(20[2-9][0-9])\b',  # Years 2020-2099
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+20[2-9][0-9]\b'
        ]
        
        dates_found = []
        for pattern in date_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dates_found.append(match.group(0))
        
        if dates_found:
            verification_data["extracted_facts"].append({
                "type": "timeline",
                "dates": dates_found
            })
        
        return verification_data


class RealTimeVerificationOrchestrator:
    """Orchestrates real-time verification for stale data updates."""
    
    def __init__(self):
        self.verifier = RealTimeFounderVerifier()
    
    async def update_stale_founder_data(
        self, 
        founder_name: str, 
        company_name: str, 
        last_update: Optional[datetime] = None,
        custom_verification_queries: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Update potentially stale founder data before ranking.
        
        Use this when:
        - Data was collected weeks/months ago  
        - About to rank founders and want current info
        - Need to verify major claims before final scoring
        """
        
        async with self.verifier:
            # Get real-time verification data
            verification_data = await self.verifier.verify_founder_data_realtime(
                founder_name, 
                company_name,
                last_update,
                custom_verification_queries
            )
            
            # Analyze verification results
            if not verification_data:
                return {
                    "updated": False,
                    "reason": "No recent information found or data is current",
                    "last_verified": datetime.now().isoformat(),
                    "confidence": 0.0
                }
            
            # Extract key updates
            key_updates = []
            funding_updates = []
            exit_updates = []
            
            for data_point in verification_data:
                if isinstance(data_point.value, dict):
                    facts = data_point.value.get("extracted_facts", [])
                    for fact in facts:
                        if fact["type"] == "funding":
                            funding_updates.append(fact)
                        elif fact["type"] == "exit":
                            exit_updates.append(fact)
                        elif fact["type"] == "valuation":
                            key_updates.append(fact)
            
            # Calculate update confidence
            update_confidence = sum(dp.confidence for dp in verification_data) / len(verification_data)
            
            return {
                "updated": True,
                "verification_data": verification_data,
                "key_updates": key_updates,
                "funding_updates": funding_updates,
                "exit_updates": exit_updates,
                "update_confidence": update_confidence,
                "last_verified": datetime.now().isoformat(),
                "data_freshness": "current" if not last_update or (datetime.now() - last_update).days < 30 else "stale"
            }
    
    async def batch_verify_stale_data(
        self,
        founder_list: List[Dict[str, Any]],  # [{"name": "...", "company": "...", "last_update": ...}]
        staleness_threshold_days: int = 30
    ) -> Dict[str, Any]:
        """Batch verify multiple founders' data that might be stale."""
        
        logger.info(f"Batch verifying {len(founder_list)} founders for stale data")
        
        verification_tasks = []
        stale_founders = []
        
        # Filter founders with stale data
        cutoff_date = datetime.now() - timedelta(days=staleness_threshold_days)
        
        for founder_info in founder_list:
            last_update = founder_info.get("last_update")
            if not last_update or last_update < cutoff_date:
                stale_founders.append(founder_info)
                
                verification_tasks.append(
                    self.update_stale_founder_data(
                        founder_info["name"],
                        founder_info["company"],
                        last_update
                    )
                )
        
        if not verification_tasks:
            logger.info("No stale data found - all founders have recent updates")
            return {
                "total_checked": len(founder_list),
                "stale_count": 0,
                "updated_count": 0,
                "results": []
            }
        
        # Execute verification tasks with rate limiting
        results = []
        for i, task in enumerate(verification_tasks):
            try:
                result = await task
                result["founder_name"] = stale_founders[i]["name"]
                result["company_name"] = stale_founders[i]["company"]
                results.append(result)
                
                # Rate limiting between requests
                if i < len(verification_tasks) - 1:
                    await asyncio.sleep(3)  # 3 second delay between verifications
                    
            except Exception as e:
                logger.error(f"Verification failed for {stale_founders[i]['name']}: {e}")
                results.append({
                    "founder_name": stale_founders[i]["name"],
                    "company_name": stale_founders[i]["company"],
                    "updated": False,
                    "error": str(e)
                })
        
        updated_count = len([r for r in results if r.get("updated", False)])
        
        logger.info(f"Batch verification complete: {updated_count}/{len(stale_founders)} updated")
        
        return {
            "total_checked": len(founder_list),
            "stale_count": len(stale_founders),
            "updated_count": updated_count,
            "staleness_threshold_days": staleness_threshold_days,
            "results": results
        }
