"""Advanced metrics extraction service for enhanced company data parsing."""

import re
import json
from typing import Dict, List, Optional, Any, Union
from decimal import Decimal
from datetime import datetime
import asyncio

from openai import AsyncOpenAI

from ..core import get_logger, RateLimiter, settings
from ..validators import validate_funding_amount, validate_year


logger = get_logger(__name__)


class MetricsExtractor:
    """Advanced extraction of funding, valuation, and company metrics from text content."""
    
    def __init__(self):
        self.openai = AsyncOpenAI(api_key=settings.openai_api_key)
        self.rate_limiter = RateLimiter(max_requests=60, time_window=60)
        
        # Funding stage patterns
        self.funding_patterns = {
            'pre_seed': r'pre[\-\s]?seed|pre[\-\s]?Series|angel|friends?\s+and\s+family',
            'seed': r'\bseed\b(?!\s+(?:stage|round))|seed\s+(?:round|funding|investment)',
            'series_a': r'series\s+a\b|series[\-\s]?a\b',
            'series_b': r'series\s+b\b|series[\-\s]?b\b',
            'series_c': r'series\s+c\b|series[\-\s]?c\b',
            'series_d': r'series\s+d\b|series[\-\s]?d\b',
            'ipo': r'\bipo\b|initial\s+public\s+offering|public\s+listing',
            'acquired': r'acquired|acquisition|bought\s+by|merger'
        }
        
        # Currency patterns
        self.currency_patterns = {
            'USD': r'\$|USD|dollars?',
            'EUR': r'€|EUR|euros?',
            'GBP': r'£|GBP|pounds?',
            'JPY': r'¥|JPY|yen'
        }
        
        # Amount multipliers
        self.multipliers = {
            'thousand': 1_000,
            'k': 1_000,
            'million': 1_000_000,
            'm': 1_000_000,
            'mn': 1_000_000,
            'billion': 1_000_000_000,
            'b': 1_000_000_000,
            'bn': 1_000_000_000,
            'trillion': 1_000_000_000_000,
            't': 1_000_000_000_000
        }
    
    async def extract_comprehensive_metrics(
        self, 
        content: str, 
        company_name: str,
        source_url: str = ""
    ) -> Dict[str, Any]:
        """Extract comprehensive company metrics from content."""
        try:
            # Run multiple extraction methods in parallel
            tasks = [
                self._extract_funding_metrics(content, company_name),
                self._extract_valuation_metrics(content, company_name),
                self._extract_employee_metrics(content),
                self._extract_growth_metrics(content),
                self._extract_competitive_metrics(content, company_name)
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine all metrics
            combined_metrics = {}
            for i, result in enumerate(results):
                if isinstance(result, dict):
                    combined_metrics.update(result)
                else:
                    logger.warning(f"Metrics extraction task {i} failed: {result}")
            
            # Add metadata
            combined_metrics.update({
                'extraction_timestamp': datetime.now().isoformat(),
                'source_url': source_url,
                'data_quality_score': self._calculate_data_quality(combined_metrics)
            })
            
            return combined_metrics
            
        except Exception as e:
            logger.error(f"Error in comprehensive metrics extraction: {e}")
            return self._get_default_metrics()
    
    async def _extract_funding_metrics(self, content: str, company_name: str) -> Dict[str, Any]:
        """Extract funding-related metrics using both regex and AI."""
        metrics = {}
        
        # Regex-based extraction first (fast)
        regex_funding = self._extract_funding_regex(content)
        
        # AI-enhanced extraction for complex cases
        ai_funding = await self._extract_funding_ai(content, company_name)
        
        # Merge and validate results
        metrics.update({
            'total_funding_usd': self._resolve_funding_amount(
                regex_funding.get('total_funding'), 
                ai_funding.get('total_funding')
            ),
            'latest_funding_usd': self._resolve_funding_amount(
                regex_funding.get('latest_round'), 
                ai_funding.get('latest_round')
            ),
            'funding_stage': self._resolve_funding_stage(
                regex_funding.get('stage'), 
                ai_funding.get('stage')
            ),
            'funding_rounds_count': ai_funding.get('rounds_count', 0),
            'last_funding_date': self._parse_funding_date(
                ai_funding.get('last_funding_date')
            ),
            'lead_investors': ai_funding.get('lead_investors', []),
            'notable_investors': ai_funding.get('notable_investors', [])
        })
        
        return {k: v for k, v in metrics.items() if v is not None}
    
    def _extract_funding_regex(self, content: str) -> Dict[str, Any]:
        """Fast regex-based funding extraction."""
        content_lower = content.lower()
        
        # Extract funding amounts
        amount_patterns = [
            r'raised?\s*\$?([0-9,\.]+)\s*(million|billion|m|b|k|thousand)',
            r'funding\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b|k|thousand)',
            r'series\s+[a-d]\s+round\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b|k|thousand)',
            r'\$([0-9,\.]+)\s*(million|billion|m|b|k|thousand)\s+(?:in\s+)?(?:series|round|funding)'
        ]
        
        amounts = []
        for pattern in amount_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for amount, unit in matches:
                try:
                    amount_clean = float(amount.replace(',', ''))
                    multiplier = self.multipliers.get(unit.lower(), 1)
                    amounts.append(amount_clean * multiplier)
                except ValueError:
                    continue
        
        # Extract funding stage
        stage = None
        for stage_name, pattern in self.funding_patterns.items():
            if re.search(pattern, content_lower, re.IGNORECASE):
                stage = stage_name.replace('_', '-')
                break
        
        return {
            'total_funding': max(amounts) if amounts else None,
            'latest_round': amounts[-1] if amounts else None,
            'stage': stage
        }
    
    async def _extract_funding_ai(self, content: str, company_name: str) -> Dict[str, Any]:
        """AI-powered funding extraction for complex cases."""
        prompt = f"""
        Analyze this content about {company_name} and extract detailed funding information.
        
        Content: {content[:2000]}
        
        Extract the following information and return as JSON:
        {{
            "total_funding": "total amount raised in USD (number only, no currency)",
            "latest_round": "most recent funding round amount in USD",
            "stage": "current funding stage (pre-seed, seed, series-a, etc.)",
            "rounds_count": "number of funding rounds",
            "last_funding_date": "date of last funding (YYYY-MM-DD if available)",
            "lead_investors": ["array of lead investor names"],
            "notable_investors": ["array of notable investor names"]
        }}
        
        Return only valid JSON. Use null for missing information.
        """
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            content_response = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if content_response.startswith("```json"):
                content_response = content_response[7:-3]
            elif content_response.startswith("```"):
                content_response = content_response[3:-3]
            
            return json.loads(content_response)
            
        except Exception as e:
            logger.debug(f"AI funding extraction failed: {e}")
            return {}
    
    async def _extract_valuation_metrics(self, content: str, company_name: str) -> Dict[str, Any]:
        """Extract valuation and market cap information."""
        content_lower = content.lower()
        
        # Regex patterns for valuation
        valuation_patterns = [
            r'valued?\s+at\s*\$?([0-9,\.]+)\s*(million|billion|m|b)',
            r'valuation\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b)',
            r'market\s+cap\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b)',
            r'worth\s*\$?([0-9,\.]+)\s*(million|billion|m|b)'
        ]
        
        valuations = []
        for pattern in valuation_patterns:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            for amount, unit in matches:
                try:
                    amount_clean = float(amount.replace(',', ''))
                    multiplier = self.multipliers.get(unit.lower(), 1)
                    valuations.append(amount_clean * multiplier)
                except ValueError:
                    continue
        
        # Look for unicorn status
        is_unicorn = bool(re.search(r'\bunicorn\b|billion.*valuation|1b.*valuation', content_lower))
        
        return {
            'current_valuation_usd': max(valuations) if valuations else None,
            'is_unicorn': is_unicorn,
            'valuation_source': 'content_analysis'
        }
    
    async def _extract_employee_metrics(self, content: str) -> Dict[str, Any]:
        """Extract employee count and growth metrics."""
        content_lower = content.lower()
        
        # Employee count patterns
        employee_patterns = [
            r'(\d+)\s+employees?',
            r'team\s+of\s+(\d+)',
            r'staff\s+of\s+(\d+)',
            r'workforce\s+of\s+(\d+)',
            r'(\d+)\s+(?:full[\-\s]?time\s+)?(?:team\s+)?members?'
        ]
        
        employee_counts = []
        for pattern in employee_patterns:
            matches = re.findall(pattern, content_lower)
            employee_counts.extend([int(match) for match in matches if match.isdigit()])
        
        # Growth indicators
        is_hiring = bool(re.search(r'hiring|recruiting|job\s+openings|careers|join\s+(?:our\s+)?team', content_lower))
        
        return {
            'employee_count': max(employee_counts) if employee_counts else None,
            'is_actively_hiring': is_hiring
        }
    
    async def _extract_growth_metrics(self, content: str) -> Dict[str, Any]:
        """Extract growth and traction metrics."""
        content_lower = content.lower()
        
        # Revenue patterns
        revenue_patterns = [
            r'revenue\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b|k)',
            r'arr\s+of\s*\$?([0-9,\.]+)\s*(million|billion|m|b|k)',
            r'annual\s+recurring\s+revenue.*?\$?([0-9,\.]+)\s*(million|billion|m|b|k)'
        ]
        
        revenues = []
        for pattern in revenue_patterns:
            matches = re.findall(pattern, content_lower)
            for amount, unit in matches:
                try:
                    amount_clean = float(amount.replace(',', ''))
                    multiplier = self.multipliers.get(unit.lower(), 1)
                    revenues.append(amount_clean * multiplier)
                except ValueError:
                    continue
        
        # Customer metrics
        customer_patterns = [
            r'(\d+)\s+(?:million\s+)?customers?',
            r'(\d+)\s+(?:million\s+)?users?',
            r'user\s+base\s+of\s+(\d+)',
            r'customer\s+base\s+of\s+(\d+)'
        ]
        
        customer_counts = []
        for pattern in customer_patterns:
            matches = re.findall(pattern, content_lower)
            customer_counts.extend([int(match) for match in matches if match.isdigit()])
        
        return {
            'annual_revenue_usd': max(revenues) if revenues else None,
            'customer_count': max(customer_counts) if customer_counts else None,
            'has_revenue_data': bool(revenues),
            'has_customer_data': bool(customer_counts)
        }
    
    async def _extract_competitive_metrics(self, content: str, company_name: str) -> Dict[str, Any]:
        """Extract competitive positioning and market metrics."""
        content_lower = content.lower()
        
        # Market position indicators
        leadership_indicators = [
            r'leading|leader\s+in|market\s+leader',
            r'first|pioneer|innovative',
            r'fastest[\-\s]growing|rapid\s+growth',
            r'award[\-\s]winning|recognized'
        ]
        
        competitive_advantages = []
        for indicator in leadership_indicators:
            if re.search(indicator, content_lower):
                competitive_advantages.append(indicator.replace(r'\s+', ' '))
        
        # Partnership indicators
        partnerships = bool(re.search(
            r'partnership|strategic\s+alliance|collaboration|integration', 
            content_lower
        ))
        
        return {
            'competitive_advantages': competitive_advantages,
            'has_strategic_partnerships': partnerships,
            'market_position_indicators': len(competitive_advantages)
        }
    
    def _resolve_funding_amount(self, regex_amount: Optional[float], ai_amount: Optional[Any]) -> Optional[float]:
        """Resolve funding amount from multiple sources."""
        amounts = []
        
        if regex_amount and validate_funding_amount(regex_amount):
            amounts.append(regex_amount)
        
        if ai_amount:
            try:
                ai_float = float(str(ai_amount).replace(',', '').replace('$', ''))
                if validate_funding_amount(ai_float):
                    amounts.append(ai_float)
            except (ValueError, TypeError):
                pass
        
        # Return the most reasonable amount
        if amounts:
            # Prefer amounts that are similar (within 20% difference)
            if len(amounts) == 2:
                ratio = max(amounts) / min(amounts)
                if ratio <= 1.2:  # Within 20%
                    return max(amounts)  # Take the higher one
            
            return max(amounts)
        
        return None
    
    def _resolve_funding_stage(self, regex_stage: Optional[str], ai_stage: Optional[str]) -> Optional[str]:
        """Resolve funding stage from multiple sources."""
        if regex_stage and ai_stage:
            # If both sources agree or are compatible
            if regex_stage == ai_stage:
                return regex_stage
            # Prefer more specific AI result
            return ai_stage
        
        return regex_stage or ai_stage
    
    def _parse_funding_date(self, date_str: Optional[str]) -> Optional[str]:
        """Parse and validate funding date."""
        if not date_str:
            return None
        
        try:
            # Try different date formats
            for fmt in ['%Y-%m-%d', '%Y-%m', '%Y']:
                try:
                    parsed_date = datetime.strptime(str(date_str), fmt)
                    if validate_year(parsed_date.year):
                        return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        except Exception:
            pass
        
        return None
    
    def _calculate_data_quality(self, metrics: Dict[str, Any]) -> float:
        """Calculate data quality score based on completeness."""
        important_fields = [
            'total_funding_usd', 'funding_stage', 'current_valuation_usd',
            'employee_count', 'annual_revenue_usd'
        ]
        
        available_fields = sum(1 for field in important_fields if metrics.get(field) is not None)
        return available_fields / len(important_fields)
    
    def _get_default_metrics(self) -> Dict[str, Any]:
        """Return default empty metrics structure."""
        return {
            'total_funding_usd': None,
            'latest_funding_usd': None,
            'funding_stage': None,
            'current_valuation_usd': None,
            'employee_count': None,
            'annual_revenue_usd': None,
            'data_quality_score': 0.0,
            'extraction_timestamp': datetime.now().isoformat()
        }
