    async def _discover_from_media_sources(
        self, 
        sources: List[CompanySource], 
        categories: Optional[List[str]], 
        regions: Optional[List[str]], 
        limit: int
    ) -> List[Company]:
        """Discover companies from real-time media monitoring sources."""
        if not sources:
            return []
            
        logger.info(f"📰 Monitoring {len(sources)} media sources...")
        
        # Use Exa for media source monitoring with targeted queries
        current_year = datetime.now().year
        queries = [
            f"AI startup funding announcement {current_year} TechCrunch VentureBeat",
            f"new artificial intelligence company launched {current_year}",
            f"startup raises seed series A funding AI {current_year}",
            f"emerging AI companies venture capital {current_year}",
            f"artificial intelligence startup news {current_year} funding",
        ]
        
        companies = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural", 
                    use_autoprompt=True,
                    num_results=limit // len(queries),
                    text={"max_characters": 2000},
                    include_domains=[s.url.split('/')[2] for s in sources if '/' in s.url]
                )
                
                for item in result.results:
                    company = await self._extract_company_data(
                        item.text, item.url, item.title
                    )
                    if company:
                        company.source_url = item.url
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Error monitoring media source with query '{query}': {e}")
                continue
        
        return companies

    async def _discover_from_accelerators(
        self, 
        sources: List[CompanySource], 
        limit: int
    ) -> List[Company]:
        """Discover companies from accelerator and demo day tracking."""
        if not sources:
            return []
            
        logger.info(f"🚀 Tracking {len(sources)} accelerator sources...")
        
        # YC, Techstars, and other accelerator-specific queries
        current_year = datetime.now().year
        queries = [
            f"Y Combinator batch {current_year} AI startups demo day",
            f"Techstars {current_year} AI artificial intelligence companies",
            f"500 Startups portfolio AI machine learning {current_year}",
            f"accelerator demo day AI startups {current_year}",
            f"YC S{current_year % 100} W{current_year % 100} AI companies",
        ]
        
        companies = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    use_autoprompt=True, 
                    num_results=limit // len(queries),
                    text={"max_characters": 2000},
                    include_domains=[
                        "ycombinator.com", "techstars.com", "500.co",
                        "angellist.com", "f6s.com", "startupgrind.com"
                    ]
                )
                
                for item in result.results:
                    company = await self._extract_company_data(
                        item.text, item.url, item.title
                    )
                    if company:
                        company.source_url = item.url
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Error tracking accelerator with query '{query}': {e}")
                continue
        
        return companies

    async def _discover_stealth_companies(
        self, 
        sources: List[CompanySource], 
        limit: int
    ) -> List[Company]:
        """Discover stealth companies through job postings and GitHub activity."""
        if not sources:
            return []
            
        logger.info(f"🕵️ Detecting stealth companies from {len(sources)} sources...")
        
        # Stealth company detection queries
        queries = [
            "stealth startup hiring AI engineers machine learning",
            "early stage AI company hiring software engineers",
            "well-funded AI startup stealth mode hiring",
            "AI startup hiring before public launch",
            "GitHub trending AI projects with significant activity",
        ]
        
        companies = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    use_autoprompt=True,
                    num_results=limit // len(queries),
                    text={"max_characters": 2000},
                    include_domains=[
                        "linkedin.com", "github.com", "jobs.lever.co",
                        "greenhouse.io", "wellfound.com", "cord.co"
                    ]
                )
                
                for item in result.results:
                    company = await self._extract_stealth_company_data(
                        item.text, item.url, item.title
                    )
                    if company:
                        company.source_url = item.url
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Error detecting stealth companies with query '{query}': {e}")
                continue
        
        return companies

    async def _discover_from_vc_sources(
        self, 
        sources: List[CompanySource], 
        limit: int
    ) -> List[Company]:
        """Discover companies from VC firm portfolio announcements."""
        if not sources:
            return []
            
        logger.info(f"💰 Monitoring {len(sources)} VC sources...")
        
        current_year = datetime.now().year
        queries = [
            f"Andreessen Horowitz a16z new portfolio company AI {current_year}",
            f"Sequoia Capital investment AI startup {current_year}",
            f"Greylock Partners portfolio AI machine learning {current_year}",
            f"First Round Capital AI investment {current_year}",
            f"VC firm announces investment AI startup {current_year}",
        ]
        
        companies = []
        for query in queries:
            try:
                await self.rate_limiter.acquire()
                
                result = self.exa.search_and_contents(
                    query,
                    type="neural",
                    use_autoprompt=True,
                    num_results=limit // len(queries),
                    text={"max_characters": 2000},
                    include_domains=[
                        "a16z.com", "sequoiacap.com", "greylock.com",
                        "firstround.com", "nea.com", "kleinerperkins.com"
                    ]
                )
                
                for item in result.results:
                    company = await self._extract_company_data(
                        item.text, item.url, item.title
                    )
                    if company:
                        company.source_url = item.url
                        companies.append(company)
                        
            except Exception as e:
                logger.error(f"Error monitoring VC source with query '{query}': {e}")
                continue
        
        return companies

    async def _extract_stealth_company_data(
        self, 
        content: str, 
        url: str, 
        title: str
    ) -> Optional[Company]:
        """Extract stealth company data with special handling for job postings."""
        prompt = f"""
Extract stealth company information from this job posting or GitHub content. Return valid JSON only:

Content: {content[:1500]}
Title: {title}

Look for:
- Company names mentioned in stealth or hiring contexts
- AI/ML focus areas from job descriptions
- Funding indicators or team size
- Location information

{{
    "name": "company name (may be unnamed/stealth)",
    "description": "what the company does based on job requirements",
    "short_description": "brief description",
    "founded_year": "estimated year or null",
    "funding_amount_millions": "estimated funding or null",
    "funding_stage": "estimated stage or null",
    "founders": ["founder names if mentioned"],
    "categories": ["AI categories from job description"],
    "city": "city name",
    "region": "state/region",
    "country": "country",
    "ai_focus": "specific AI area from job posting",
    "sector": "business sector",
    "website": "company website if mentioned",
    "is_stealth": true
}}
"""
        
        try:
            await self.rate_limiter.acquire()
            
            response = await self.openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            raw_content = response.choices[0].message.content.strip()
            
            # Clean JSON response
            if raw_content.startswith("```json"):
                raw_content = raw_content[7:]
            if raw_content.startswith("```"):
                raw_content = raw_content[3:]
            if raw_content.endswith("```"):
                raw_content = raw_content[:-3]
            
            result = json.loads(raw_content.strip())
            
            # Only create company if we have meaningful data
            if not result.get("name") or result["name"].lower() in ["stealth", "unnamed", "unknown"]:
                return None
            
            # Map funding stage
            funding_stage = None
            if result.get("funding_stage"):
                try:
                    funding_stage = FundingStage(result["funding_stage"])
                except ValueError:
                    funding_stage = FundingStage.UNKNOWN
            
            # Convert funding
            funding_millions = result.get("funding_amount_millions")
            funding_total_usd = None
            if funding_millions and isinstance(funding_millions, (int, float)):
                funding_total_usd = funding_millions * 1_000_000
            
            company = Company(
                uuid=f"comp_stealth_{hash(result.get('name', ''))}",
                name=clean_text(result.get("name", "")),
                description=clean_text(result.get("description", "")),
                short_description=clean_text(result.get("short_description", "")),
                founded_year=result.get("founded_year"),
                funding_total_usd=funding_total_usd,
                funding_stage=funding_stage,
                founders=result.get("founders", []),
                categories=result.get("categories", []),
                city=clean_text(result.get("city", "")),
                region=clean_text(result.get("region", "")),
                country=clean_text(result.get("country", "")),
                ai_focus=clean_text(result.get("ai_focus", "")),
                sector=clean_text(result.get("sector", "")),
                website=result.get("website"),
                source_url=url,
                extraction_date=datetime.utcnow()
            )
            
            return company if company.name else None
            
        except Exception as e:
            logger.error(f"Error extracting stealth company data: {e}")
            return None
    
    # Keep the original find_companies method for backward compatibility
    async def find_companies(
        self, 
        limit: int = 50,
        categories: Optional[List[str]] = None,
        regions: Optional[List[str]] = None
    ) -> List[Company]:
        """Legacy method - use discover_companies for full functionality."""
        return await self.discover_companies(limit, categories, regions)
