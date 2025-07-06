"""Structured prompts for L1-L10 founder classification using Claude Sonnet 4."""

from typing import Dict, Any
from .models import FounderProfile


class RankingPrompts:
    """Centralized prompt templates for founder ranking."""
    
    SYSTEM_PROMPT = """You are an expert VC analyst specializing in founder assessment using the L1-L10 Carnegie Mellon research-based framework. Your task is to classify founders based on their experience level and track record.

L1-L10 CLASSIFICATION FRAMEWORK:

L10 - Legendary Entrepreneurs
- Multiple IPOs or exits >$1B
- Created entire industries or market categories
- Examples: Jobs, Gates, Bezos, Musk

L9 - Transformational Leaders  
- 1 major IPO or exit >$1B, building second major company
- Recognized as industry visionaries
- Examples: Marc Benioff (Salesforce), Drew Houston (Dropbox) building next

L8 - Proven Unicorn Builders
- Built 1+ companies to $1B+ valuation
- Examples: Brian Chesky (Airbnb), Daniel Ek (Spotify)

L7 - Elite Serial Entrepreneurs
- DEFINITION: Founders who have achieved multiple large-scale exits or built multiple unicorn companies
- KEY CRITERIA: 2+ companies with exits >$100M OR 2+ unicorn companies founded
- VERIFICATION: Press releases about exits, SEC filings, Crunchbase exit data, Forbes/TechCrunch coverage of major exits
- SEARCH PATTERNS: "founder name + IPO", "founder name + acquisition + $100M+", "founder name + unicorn founder"
- Examples: Multiple IPOs, consistent pattern of scaling companies to >$1B valuation

L6 - Market Innovators and Thought Leaders  
- DEFINITION: Founders recognized for groundbreaking innovation who have disrupted or created new markets
- KEY CRITERIA: Significant media recognition as innovator OR multiple patents OR major industry awards
- VERIFICATION: Patent databases (USPTO, Google Patents), award listings, speaking engagements, major media profiles
- SEARCH PATTERNS: "founder name + innovation award", "founder name + thought leader", "founder name + Forbes", "founder name + TED talk"
- Examples: Patents, industry awards, keynote speeches, book publications, media recognition as innovator

L5 - Growth-Stage Entrepreneurs
- DEFINITION: Founders who have scaled companies to significant funding levels (>$50M) and are positioned for major exits
- KEY CRITERIA: Led company that raised >$50M OR company preparing for IPO/major exit
- VERIFICATION: Crunchbase funding data, SEC filings, venture capital database entries, press releases
- SEARCH PATTERNS: "founder name + Series C", "founder name + $50M funding", "founder name + IPO preparation"
- Examples: Companies that raised >$50M, late-stage funding rounds, IPO filings, acquisition rumors >$100M

L4 - Proven Operators with Exits or Executive Experience
- DEFINITION: Founders with successful small-to-medium exits or significant executive roles at notable tech companies
- KEY CRITERIA: Exit between $10M-$100M OR C-level/VP role at notable tech company (>1000 employees)
- VERIFICATION: LinkedIn employment history, company websites, acquisition announcements, Crunchbase executive data
- SEARCH PATTERNS: "founder name + exit", "founder name + CTO", "founder name + VP", "founder name + acquired"
- Examples: Executive titles (C-level, VP) at known tech companies, small-medium exits, senior roles at unicorns

L3 - Technical and Management Veterans
- DEFINITION: Individuals with 10-15 years of technical and management experience, including senior roles at high-growth companies
- KEY CRITERIA: 10+ years combined technical/management experience OR PhD in relevant field OR senior role at fast-growing company
- VERIFICATION: University records, LinkedIn work history, patent filings, academic publications
- SEARCH PATTERNS: "founder name + PhD", "founder name + senior engineer", "founder name + 10 years experience"
- Examples: PhD degrees, senior technical roles, management positions, 10+ years industry experience, patents

L2 - Early-Stage Entrepreneurs
- DEFINITION: Founders with limited startup experience or accelerator backgrounds showing early promise
- KEY CRITERIA: Accelerator graduate OR 2-5 years startup experience OR seed funding raised
- VERIFICATION: Accelerator websites, demo day listings, seed funding announcements, university graduation records
- SEARCH PATTERNS: "founder name + Y Combinator", "founder name + Techstars", "founder name + accelerator", "founder name + seed funding"
- Examples: Accelerator participation, seed funding rounds, early-stage company experience, recent graduation from top programs

L1 - Nascent Founders with Potential
- DEFINITION: New entrepreneurs with minimal experience but demonstrating ambition and potential
- KEY CRITERIA: <2 years professional experience OR first-time founder OR recent graduate (<3 years)
- VERIFICATION: University graduation dates, LinkedIn work history duration, company founding dates
- SEARCH PATTERNS: "founder name + recent graduate", "founder name + first startup", "founder name + young entrepreneur"
- Examples: Recent university graduation, first-time founder, limited work experience, early-stage startup

CRITICAL ANALYSIS REQUIREMENTS:
1. Focus on CONCRETE ACHIEVEMENTS, not just titles
2. Verify claims with cross-references when possible
3. Be conservative - err on the side of lower classification if uncertain
4. Look for financial outcomes (exits, valuations, funding amounts)
5. Consider company scale and impact, not just personal roles
6. Prioritize entrepreneurial track record over employment history
7. Account for age and career stage appropriately

CONFIDENCE SCORING:
- Only classify with confidence ≥0.75 (75%) for actionable results
- If confidence <0.75, return "INSUFFICIENT_DATA" with reasoning
- Higher levels (L6+) require stronger evidence and higher confidence
- Consider data quality and verification possibilities in confidence scoring

VERIFICATION HIERARCHY:
- Start with highest level (L7) and work down
- Look for multiple confirming sources for major claims
- Use financial thresholds as primary discriminators
- Cross-reference timeline data to ensure accuracy
- Prioritize official sources (SEC, university records) over media reports

OUTPUT FORMAT:
Provide a structured JSON response with:
- level: L1-L10 classification OR "INSUFFICIENT_DATA" if confidence <0.75
- confidence_score: 0.0-1.0 (how certain you are)
- reasoning: Detailed explanation for the classification
- evidence: List of specific achievements/facts supporting the level
- verification_sources: What additional sources would verify key claims"""

    @staticmethod
    def create_founder_analysis_prompt(profile: FounderProfile, company_founding_year: int = None) -> str:
        """Create the main analysis prompt for a founder with year context."""
        
        # Build experience summary
        experiences = []
        if profile.experience_1_title and profile.experience_1_company:
            experiences.append(f"{profile.experience_1_title} at {profile.experience_1_company}")
        if profile.experience_2_title and profile.experience_2_company:
            experiences.append(f"{profile.experience_2_title} at {profile.experience_2_company}")
        if profile.experience_3_title and profile.experience_3_company:
            experiences.append(f"{profile.experience_3_title} at {profile.experience_3_company}")
        
        experience_text = "; ".join(experiences) if experiences else "No detailed experience listed"
        
        # Build education summary
        education = []
        if profile.education_1_school and profile.education_1_degree:
            education.append(f"{profile.education_1_degree} from {profile.education_1_school}")
        if profile.education_2_school and profile.education_2_degree:
            education.append(f"{profile.education_2_degree} from {profile.education_2_school}")
        
        education_text = "; ".join(education) if education else "No education details listed"
        
        # Build skills summary
        skills = [skill for skill in [profile.skill_1, profile.skill_2, profile.skill_3] if skill]
        skills_text = ", ".join(skills) if skills else "No skills listed"
        
        # Add year context if available
        year_context = ""
        if company_founding_year:
            year_context = f"""
YEAR CONTEXT:
This founder's current company was founded in {company_founding_year}. 
Consider their experience level relative to building a company in {company_founding_year}.
Evaluate their track record and preparation leading up to {company_founding_year}.
"""
        
        prompt = f"""
FOUNDER ANALYSIS REQUEST

Analyze this founder and classify them using the L1-L10 framework:

FOUNDER PROFILE:
Name: {profile.name}
Current Company: {profile.company_name}
Current Title: {profile.title}
Location: {profile.location or 'Not specified'}
Estimated Age: {profile.estimated_age or 'Not specified'}

LinkedIn: {profile.linkedin_url or 'Not provided'}

ABOUT/BIO:
{profile.about or 'No bio provided'}

EXPERIENCE HISTORY:
{experience_text}

EDUCATION:
{education_text}

SKILLS:
{skills_text}
{year_context}
ANALYSIS INSTRUCTIONS:
1. Carefully evaluate this founder's track record using the L1-L10 framework
2. Look for concrete evidence of exits, valuations, company scale, funding rounds
3. Consider their current company context and role
4. Factor in their estimated age and career progression
5. Be conservative - require solid evidence for higher levels (L6+)
6. For levels L7+, look for multiple successful companies or major exits
7. For levels L4-L6, focus on proven execution and scale achievements
8. For levels L1-L3, assess potential based on education, early experience, and current trajectory
9. If year context is provided, evaluate their readiness and experience level for founding a company in that specific year

Key questions to address:
- What concrete achievements can be verified?
- What scale of companies have they built or been part of?
- Any evidence of exits, major funding rounds, or valuations?
- How does their experience compare to the L1-L10 definitions?
- What level of risk/uncertainty exists in this assessment?

Return your analysis as a valid JSON object with the exact structure:
{{
    "level": "L1", 
    "confidence_score": 0.85,
    "reasoning": "Detailed explanation of why this founder fits this level...",
    "evidence": ["Specific fact 1", "Specific fact 2", "Specific fact 3"],
    "verification_sources": ["LinkedIn profile verification", "Company funding data", "Exit records"]
}}
"""
        return prompt

    @staticmethod
    def create_batch_analysis_prompt(profiles: list[FounderProfile]) -> str:
        """Create prompt for analyzing multiple founders in batch."""
        
        founders_summary = []
        for i, profile in enumerate(profiles, 1):
            founders_summary.append(
                f"FOUNDER {i}: {profile.name} - {profile.title} at {profile.company_name}"
            )
        
        prompt = f"""
BATCH FOUNDER ANALYSIS REQUEST

Analyze these {len(profiles)} founders and classify each using the L1-L10 framework:

FOUNDERS TO ANALYZE:
{chr(10).join(founders_summary)}

For each founder, provide the same detailed analysis as you would individually. 
Return a JSON array where each object represents one founder's analysis.

Expected format:
[
    {{
        "founder_name": "Name",
        "level": "L2",
        "confidence_score": 0.75,
        "reasoning": "...",
        "evidence": ["..."],
        "verification_sources": ["..."]
    }},
    ...
]
"""
        return prompt

    @staticmethod
    def create_verification_prompt(profile: FounderProfile, initial_classification: str) -> str:
        """Create prompt for verifying and refining initial classification."""
        
        prompt = f"""
FOUNDER CLASSIFICATION VERIFICATION

Initial classification: {initial_classification}
Founder: {profile.name} at {profile.company_name}

Please verify this classification by:

1. Double-checking the evidence against L1-L10 criteria
2. Looking for any missed indicators that could change the level
3. Assessing if the confidence score should be adjusted
4. Identifying the most critical verification needs

Focus particularly on:
- Financial outcomes (exits, valuations, funding)
- Company scale and growth metrics  
- Founder's specific contributions vs. just being present
- Industry impact and recognition
- Track record consistency

Return refined classification with same JSON structure, noting any changes made and why.
"""
        return prompt

    @staticmethod
    def get_level_descriptions() -> Dict[str, str]:
        """Get detailed descriptions for each level."""
        return {
            "L10": "Legendary Entrepreneurs - Multiple IPOs or exits >$1B, created entire industries",
            "L9": "Transformational Leaders - 1 major IPO/exit >$1B, building second major company", 
            "L8": "Proven Unicorn Builders - Built 1+ companies to $1B+ valuation",
            "L7": "Elite Serial Entrepreneurs - 2+ exits >$100M OR 2+ unicorn companies",
            "L6": "Market Innovators and Thought Leaders - Groundbreaking innovation, disrupted markets",
            "L5": "Growth-Stage Entrepreneurs - Scaled companies to >$50M funding, positioned for major exits",
            "L4": "Proven Operators - Small-medium exits ($10M-$100M) OR C-level at notable tech companies",
            "L3": "Technical and Management Veterans - 10+ years experience OR PhD in relevant field",
            "L2": "Early-Stage Entrepreneurs - Accelerator graduate OR 2-5 years startup experience",
            "L1": "Nascent Founders - <2 years experience OR first-time founder OR recent graduate"
        }
