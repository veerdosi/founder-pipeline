"""University education verification for founder PhD tracking (L3 criteria)."""

import asyncio
import aiohttp
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import quote, urljoin
import json

from ...core import get_logger


logger = get_logger(__name__)


@dataclass
class EducationRecord:
    """Individual education record."""
    institution: str
    degree_type: str  # "PhD", "MS", "BS", "MBA", etc.
    field_of_study: str
    graduation_year: Optional[int] = None
    verification_status: str = "unverified"  # "verified", "likely", "unverified", "disputed"
    verification_sources: List[str] = field(default_factory=list)
    confidence_score: float = 0.0  # 0-1 based on source quality
    additional_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class UniversityProfile:
    """University profile data for verification."""
    university_name: str
    official_domains: List[str] = field(default_factory=list)
    faculty_directory_url: Optional[str] = None
    alumni_directory_url: Optional[str] = None
    phd_programs: List[str] = field(default_factory=list)
    ranking_tier: str = "unknown"  # "top_10", "top_50", "top_100", "other"


@dataclass
class FounderEducationProfile:
    """Complete education profile for a founder."""
    founder_name: str
    education_records: List[EducationRecord] = field(default_factory=list)
    phd_degrees: List[EducationRecord] = field(default_factory=list)
    advanced_degrees: List[EducationRecord] = field(default_factory=list)
    total_years_education: int = 0
    highest_degree: Optional[str] = None
    technical_field_background: bool = False
    top_tier_institution: bool = False
    academic_publications: List[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def calculate_education_metrics(self):
        """Calculate derived education metrics."""
        if not self.education_records:
            return
        
        # Extract PhD degrees
        self.phd_degrees = [
            record for record in self.education_records 
            if "phd" in record.degree_type.lower() or "ph.d" in record.degree_type.lower()
        ]
        
        # Extract advanced degrees (Masters+)
        self.advanced_degrees = [
            record for record in self.education_records
            if record.degree_type.upper() in ["MS", "MA", "MBA", "PHD", "MD", "JD", "PHD"]
        ]
        
        # Determine highest degree
        degree_hierarchy = ["PHD", "MD", "JD", "MBA", "MS", "MA", "BS", "BA"]
        for degree in degree_hierarchy:
            if any(degree.lower() in record.degree_type.lower() for record in self.education_records):
                self.highest_degree = degree
                break
        
        # Check for technical field background
        technical_fields = [
            "computer science", "engineering", "physics", "mathematics", "chemistry", 
            "biology", "artificial intelligence", "machine learning", "data science"
        ]
        self.technical_field_background = any(
            any(field.lower() in record.field_of_study.lower() for field in technical_fields)
            for record in self.education_records
        )
        
        # Check for top-tier institutions
        top_tier_universities = [
            "mit", "stanford", "harvard", "caltech", "berkeley", "cmu", "princeton", 
            "yale", "columbia", "cornell", "penn", "chicago", "northwestern"
        ]
        self.top_tier_institution = any(
            any(uni in record.institution.lower() for uni in top_tier_universities)
            for record in self.education_records
        )
        
        # Estimate total years of education (rough calculation)
        degree_years = {"BS": 4, "BA": 4, "MS": 2, "MA": 2, "MBA": 2, "PHD": 5, "MD": 4, "JD": 3}
        self.total_years_education = sum(
            degree_years.get(record.degree_type.upper(), 2) 
            for record in self.education_records
        )
    
    def meets_l3_criteria(self) -> Dict[str, bool]:
        """Check if founder meets L3 education criteria."""
        return {
            "has_phd": len(self.phd_degrees) > 0,
            "advanced_degree": len(self.advanced_degrees) > 0,
            "technical_background": self.technical_field_background,
            "top_tier_institution": self.top_tier_institution,
            "verified_education": any(
                record.verification_status == "verified" 
                for record in self.education_records
            )
        }


class UniversityDataCollector:
    """Collects and verifies university education data for founders."""
    
    def __init__(self):
        self.session = None
        self.rate_limit_delay = 1.0
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
        }
        
        # Top universities with known directory patterns
        self.university_profiles = self._load_university_profiles()
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def verify_education(
        self, 
        founder_name: str, 
        claimed_degrees: List[Dict[str, str]] = None
    ) -> FounderEducationProfile:
        """Verify education credentials for a founder."""
        logger.info(f"🎓 Verifying education for {founder_name}")
        
        profile = FounderEducationProfile(founder_name=founder_name)
        
        try:
            # If specific degrees are claimed, verify them
            if claimed_degrees:
                for degree_claim in claimed_degrees:
                    verified_record = await self._verify_specific_degree(
                        founder_name, degree_claim
                    )
                    if verified_record:
                        profile.education_records.append(verified_record)
            
            # Search academic databases for publications (PhD indicator)
            publications = await self._search_academic_publications(founder_name)
            profile.academic_publications = publications
            
            # Search Google Scholar for academic profile
            scholar_education = await self._search_google_scholar(founder_name)
            profile.education_records.extend(scholar_education)
            
            # Remove duplicates and calculate metrics
            profile.education_records = self._deduplicate_education_records(profile.education_records)
            profile.calculate_education_metrics()
            
            logger.info(f"✅ Found {len(profile.education_records)} education records for {founder_name}")
            return profile
            
        except Exception as e:
            logger.error(f"Error verifying education for {founder_name}: {e}")
            return profile
    
    async def _verify_specific_degree(
        self, 
        founder_name: str, 
        degree_claim: Dict[str, str]
    ) -> Optional[EducationRecord]:
        """Verify a specific degree claim against university records."""
        institution = degree_claim.get("institution", "")
        degree_type = degree_claim.get("degree", "")
        field = degree_claim.get("field", "")
        year = degree_claim.get("year")
        
        if not institution or not degree_type:
            return None
        
        try:
            # Get university profile
            uni_profile = self._find_university_profile(institution)
            if not uni_profile:
                return self._create_unverified_record(degree_claim, founder_name)
            
            # Try to verify through faculty directory (for PhD)
            if "phd" in degree_type.lower():
                faculty_verification = await self._check_faculty_directory(
                    founder_name, uni_profile, field
                )
                if faculty_verification:
                    return EducationRecord(
                        institution=institution,
                        degree_type=degree_type,
                        field_of_study=field,
                        graduation_year=int(year) if year else None,
                        verification_status="verified",
                        verification_sources=["faculty_directory"],
                        confidence_score=0.9,
                        additional_info={"verification_method": "faculty_directory"}
                    )
            
            # Try to verify through alumni directory
            alumni_verification = await self._check_alumni_directory(
                founder_name, uni_profile, degree_type, year
            )
            if alumni_verification:
                return EducationRecord(
                    institution=institution,
                    degree_type=degree_type,
                    field_of_study=field,
                    graduation_year=int(year) if year else None,
                    verification_status="likely",
                    verification_sources=["alumni_directory"],
                    confidence_score=0.7
                )
            
            # Fallback to creating likely record if top-tier institution
            if uni_profile.ranking_tier in ["top_10", "top_50"]:
                return EducationRecord(
                    institution=institution,
                    degree_type=degree_type,
                    field_of_study=field,
                    graduation_year=int(year) if year else None,
                    verification_status="likely",
                    verification_sources=["institution_ranking"],
                    confidence_score=0.5
                )
            
            return self._create_unverified_record(degree_claim, founder_name)
            
        except Exception as e:
            logger.warning(f"Error verifying degree at {institution}: {e}")
            return self._create_unverified_record(degree_claim, founder_name)
    
    async def _search_academic_publications(self, founder_name: str) -> List[str]:
        """Search academic databases for publications (PhD indicator)."""
        publications = []
        
        try:
            # Search arXiv for academic papers
            arxiv_papers = await self._search_arxiv(founder_name)
            publications.extend(arxiv_papers)
            
            # Search PubMed for life sciences papers
            pubmed_papers = await self._search_pubmed(founder_name)
            publications.extend(pubmed_papers)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Error searching academic publications for {founder_name}: {e}")
        
        return publications[:10]  # Limit to 10 most relevant
    
    async def _search_arxiv(self, founder_name: str) -> List[str]:
        """Search arXiv for academic papers."""
        try:
            # arXiv API search
            name_query = quote(founder_name)
            url = f"http://export.arxiv.org/api/query?search_query=au:{name_query}&max_results=5"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Parse XML response
                    papers = []
                    if "<title>" in content and "<entry>" in content:
                        # Extract paper titles (simplified parsing)
                        title_matches = re.findall(r'<title>(.*?)</title>', content)
                        papers = [title.strip() for title in title_matches[1:6]]  # Skip first (query info)
                    
                    return papers
            
        except Exception as e:
            logger.warning(f"arXiv search failed for {founder_name}: {e}")
        
        return []
    
    async def _search_pubmed(self, founder_name: str) -> List[str]:
        """Search PubMed for life sciences papers."""
        try:
            # PubMed API search
            name_query = quote(f'"{founder_name}"[Author]')
            url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={name_query}&retmax=5"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Extract PMIDs
                    pmid_matches = re.findall(r'<Id>(\d+)</Id>', content)
                    
                    # Get paper titles for PMIDs
                    if pmid_matches:
                        pmids = ",".join(pmid_matches[:3])
                        summary_url = f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi?db=pubmed&id={pmids}"
                        
                        async with self.session.get(summary_url) as summary_response:
                            if summary_response.status == 200:
                                summary_content = await summary_response.text()
                                titles = re.findall(r'<Item Name="Title" Type="String">(.*?)</Item>', summary_content)
                                return titles
            
        except Exception as e:
            logger.warning(f"PubMed search failed for {founder_name}: {e}")
        
        return []
    
    async def _search_google_scholar(self, founder_name: str) -> List[EducationRecord]:
        """Search Google Scholar for academic profile indicators."""
        education_records = []
        
        try:
            # Google Scholar search (simplified approach)
            search_query = quote(f'"{founder_name}" author')
            url = f"https://scholar.google.com/scholar?q={search_query}&hl=en"
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Look for university affiliations in results
                    university_patterns = [
                        r'(University of [A-Za-z\s]+)',
                        r'([A-Za-z\s]+ University)',
                        r'(MIT|Stanford|Harvard|Caltech|Berkeley)',
                        r'([A-Za-z\s]+ Institute of Technology)'
                    ]
                    
                    found_universities = set()
                    for pattern in university_patterns:
                        matches = re.findall(pattern, content)
                        found_universities.update(matches)
                    
                    # Create education records for found affiliations
                    for university in list(found_universities)[:3]:
                        if len(university.strip()) > 3:  # Filter out short matches
                            record = EducationRecord(
                                institution=university.strip(),
                                degree_type="Unknown",
                                field_of_study="Unknown",
                                verification_status="likely",
                                verification_sources=["google_scholar"],
                                confidence_score=0.4,
                                additional_info={"source": "academic_affiliation"}
                            )
                            education_records.append(record)
            
            await asyncio.sleep(self.rate_limit_delay)
            
        except Exception as e:
            logger.warning(f"Google Scholar search failed for {founder_name}: {e}")
        
        return education_records
    
    def _find_university_profile(self, institution_name: str) -> Optional[UniversityProfile]:
        """Find university profile by name."""
        institution_lower = institution_name.lower()
        
        for profile in self.university_profiles:
            if (profile.university_name.lower() in institution_lower or 
                institution_lower in profile.university_name.lower()):
                return profile
        
        return None
    
    def _load_university_profiles(self) -> List[UniversityProfile]:
        """Load comprehensive university profiles for verification."""
        return [
            # Top 10 US Universities
            UniversityProfile(
                university_name="Massachusetts Institute of Technology",
                official_domains=["mit.edu"],
                faculty_directory_url="https://web.mit.edu/faculty/",
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Physics", "Mathematics", "Biology"]
            ),
            UniversityProfile(
                university_name="Stanford University", 
                official_domains=["stanford.edu"],
                faculty_directory_url="https://faculty.stanford.edu/",
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Business", "Medicine", "Physics"]
            ),
            UniversityProfile(
                university_name="Harvard University",
                official_domains=["harvard.edu"],
                faculty_directory_url="https://www.harvard.edu/faculty/",
                ranking_tier="top_10",
                phd_programs=["Medicine", "Law", "Business", "Computer Science", "Biology"]
            ),
            UniversityProfile(
                university_name="California Institute of Technology",
                official_domains=["caltech.edu"],
                ranking_tier="top_10",
                phd_programs=["Physics", "Engineering", "Chemistry", "Biology", "Mathematics"]
            ),
            UniversityProfile(
                university_name="University of California Berkeley",
                official_domains=["berkeley.edu"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Physics", "Chemistry", "Economics"]
            ),
            UniversityProfile(
                university_name="Carnegie Mellon University",
                official_domains=["cmu.edu"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Robotics", "Machine Learning"]
            ),
            UniversityProfile(
                university_name="Princeton University",
                official_domains=["princeton.edu"],
                ranking_tier="top_10",
                phd_programs=["Physics", "Mathematics", "Computer Science", "Economics", "Engineering"]
            ),
            UniversityProfile(
                university_name="Yale University",
                official_domains=["yale.edu"],
                ranking_tier="top_10",
                phd_programs=["Medicine", "Law", "Computer Science", "Biology", "Physics"]
            ),
            UniversityProfile(
                university_name="University of Chicago",
                official_domains=["uchicago.edu"],
                ranking_tier="top_10",
                phd_programs=["Economics", "Physics", "Mathematics", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="Columbia University",
                official_domains=["columbia.edu"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Business", "Physics"]
            ),
            
            # Top 50 US Universities
            UniversityProfile(
                university_name="Cornell University",
                official_domains=["cornell.edu"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Agriculture", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="University of Pennsylvania",
                official_domains=["upenn.edu"],
                ranking_tier="top_50",
                phd_programs=["Business", "Medicine", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Northwestern University",
                official_domains=["northwestern.edu"],
                ranking_tier="top_50",
                phd_programs=["Business", "Medicine", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Johns Hopkins University",
                official_domains=["jhu.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Public Health", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Duke University",
                official_domains=["duke.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Business", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Dartmouth College",
                official_domains=["dartmouth.edu"],
                ranking_tier="top_50",
                phd_programs=["Business", "Medicine", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Brown University",
                official_domains=["brown.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Medicine", "Engineering", "Applied Mathematics"]
            ),
            UniversityProfile(
                university_name="Vanderbilt University",
                official_domains=["vanderbilt.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Business", "Education"]
            ),
            UniversityProfile(
                university_name="Rice University",
                official_domains=["rice.edu"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Business", "Medicine"]
            ),
            UniversityProfile(
                university_name="Washington University in St. Louis",
                official_domains=["wustl.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Business", "Engineering", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Georgetown University",
                official_domains=["georgetown.edu"],
                ranking_tier="top_50",
                phd_programs=["Law", "Medicine", "Business", "Public Policy"]
            ),
            UniversityProfile(
                university_name="University of Notre Dame",
                official_domains=["nd.edu"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Business", "Computer Science", "Physics"]
            ),
            UniversityProfile(
                university_name="Emory University",
                official_domains=["emory.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Business", "Public Health", "Computer Science"]
            ),
            UniversityProfile(
                university_name="University of California Los Angeles",
                official_domains=["ucla.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="University of Southern California",
                official_domains=["usc.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Business", "Medicine"]
            ),
            UniversityProfile(
                university_name="New York University",
                official_domains=["nyu.edu"],
                ranking_tier="top_50",
                phd_programs=["Business", "Medicine", "Computer Science", "Law"]
            ),
            UniversityProfile(
                university_name="Tufts University",
                official_domains=["tufts.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "International Relations"]
            ),
            UniversityProfile(
                university_name="Wake Forest University",
                official_domains=["wfu.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Business", "Law", "Computer Science"]
            ),
            UniversityProfile(
                university_name="University of Virginia",
                official_domains=["virginia.edu"],
                ranking_tier="top_50",
                phd_programs=["Business", "Medicine", "Law", "Engineering"]
            ),
            UniversityProfile(
                university_name="University of Michigan",
                official_domains=["umich.edu"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Business", "Medicine", "Computer Science"]
            ),
            UniversityProfile(
                university_name="University of North Carolina at Chapel Hill",
                official_domains=["unc.edu"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Business", "Public Health", "Computer Science"]
            ),
            
            # International Top Universities
            UniversityProfile(
                university_name="University of Oxford",
                official_domains=["ox.ac.uk"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Medicine", "Physics", "Mathematics", "Engineering"]
            ),
            UniversityProfile(
                university_name="University of Cambridge",
                official_domains=["cam.ac.uk"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Mathematics", "Physics", "Medicine"]
            ),
            UniversityProfile(
                university_name="Imperial College London",
                official_domains=["imperial.ac.uk"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Medicine", "Physics", "Chemistry"]
            ),
            UniversityProfile(
                university_name="University College London",
                official_domains=["ucl.ac.uk"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Architecture"]
            ),
            UniversityProfile(
                university_name="London School of Economics",
                official_domains=["lse.ac.uk"],
                ranking_tier="top_50",
                phd_programs=["Economics", "Finance", "Political Science", "Management"]
            ),
            UniversityProfile(
                university_name="ETH Zurich",
                official_domains=["ethz.ch"],
                ranking_tier="top_10",
                phd_programs=["Computer Science", "Engineering", "Physics", "Mathematics", "Chemistry"]
            ),
            UniversityProfile(
                university_name="École Polytechnique Fédérale de Lausanne",
                official_domains=["epfl.ch"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Life Sciences"]
            ),
            UniversityProfile(
                university_name="University of Toronto",
                official_domains=["utoronto.ca"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="McGill University",
                official_domains=["mcgill.ca"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="University of British Columbia",
                official_domains=["ubc.ca"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="Australian National University",
                official_domains=["anu.edu.au"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Physics", "Engineering", "Economics"]
            ),
            UniversityProfile(
                university_name="University of Melbourne",
                official_domains=["unimelb.edu.au"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="University of Sydney",
                official_domains=["sydney.edu.au"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="National University of Singapore",
                official_domains=["nus.edu.sg"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="Nanyang Technological University",
                official_domains=["ntu.edu.sg"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Business", "Materials Science"]
            ),
            UniversityProfile(
                university_name="University of Hong Kong",
                official_domains=["hku.hk"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Engineering", "Computer Science", "Business"]
            ),
            UniversityProfile(
                university_name="Hong Kong University of Science and Technology",
                official_domains=["ust.hk"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Business", "Science"]
            ),
            UniversityProfile(
                university_name="Technical University of Munich",
                official_domains=["tum.de"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Physics", "Chemistry"]
            ),
            UniversityProfile(
                university_name="RWTH Aachen University",
                official_domains=["rwth-aachen.de"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Physics", "Materials Science"]
            ),
            UniversityProfile(
                university_name="École Polytechnique",
                official_domains=["polytechnique.edu"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Physics", "Mathematics", "Computer Science"]
            ),
            UniversityProfile(
                university_name="Sorbonne University",
                official_domains=["sorbonne-universite.fr"],
                ranking_tier="top_50",
                phd_programs=["Medicine", "Physics", "Computer Science", "Mathematics"]
            ),
            UniversityProfile(
                university_name="University of Tokyo",
                official_domains=["u-tokyo.ac.jp"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Medicine", "Physics"]
            ),
            UniversityProfile(
                university_name="Kyoto University",
                official_domains=["kyoto-u.ac.jp"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Medicine", "Science", "Agriculture"]
            ),
            UniversityProfile(
                university_name="Seoul National University",
                official_domains=["snu.ac.kr"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="KAIST",
                official_domains=["kaist.ac.kr"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Chemistry"]
            ),
            UniversityProfile(
                university_name="Tsinghua University",
                official_domains=["tsinghua.edu.cn"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Physics", "Economics"]
            ),
            UniversityProfile(
                university_name="Peking University",
                official_domains=["pku.edu.cn"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Physics", "Mathematics", "Economics"]
            ),
            UniversityProfile(
                university_name="Shanghai Jiao Tong University",
                official_domains=["sjtu.edu.cn"],
                ranking_tier="top_50",
                phd_programs=["Engineering", "Computer Science", "Medicine", "Business"]
            ),
            UniversityProfile(
                university_name="Fudan University",
                official_domains=["fudan.edu.cn"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Medicine", "Economics", "Physics"]
            ),
            
            # Top Indian Institutes
            UniversityProfile(
                university_name="Indian Institute of Technology Bombay",
                official_domains=["iitb.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Mathematics"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Technology Delhi",
                official_domains=["iitd.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Mathematics"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Technology Madras",
                official_domains=["iitm.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Ocean Engineering"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Technology Kanpur",
                official_domains=["iitk.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Mathematics"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Technology Kharagpur",
                official_domains=["iitkgp.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Management", "Architecture"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Science",
                official_domains=["iisc.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Physics", "Chemistry", "Biology", "Engineering"]
            ),
            UniversityProfile(
                university_name="Indian Institute of Management Ahmedabad",
                official_domains=["iima.ac.in"],
                ranking_tier="top_50",
                phd_programs=["Management", "Economics", "Public Policy"]
            ),
            
            # Other Notable Technical Universities
            UniversityProfile(
                university_name="Georgia Institute of Technology",
                official_domains=["gatech.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Industrial Engineering"]
            ),
            UniversityProfile(
                university_name="University of Illinois Urbana-Champaign",
                official_domains=["illinois.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Physics", "Business"]
            ),
            UniversityProfile(
                university_name="University of Texas at Austin",
                official_domains=["utexas.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Business", "Medicine"]
            ),
            UniversityProfile(
                university_name="University of Washington",
                official_domains=["washington.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Medicine", "Engineering", "Public Health"]
            ),
            UniversityProfile(
                university_name="University of California San Diego",
                official_domains=["ucsd.edu"],
                ranking_tier="top_50",
                phd_programs=["Computer Science", "Engineering", "Medicine", "Biology"]
            )
        ]
    
    async def _check_faculty_directory(
        self, 
        founder_name: str, 
        uni_profile: UniversityProfile, 
        field: str
    ) -> bool:
        """Check if founder appears in faculty directory."""
        if not uni_profile.faculty_directory_url:
            return False
        
        try:
            async with self.session.get(uni_profile.faculty_directory_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return founder_name.lower() in content.lower()
        except:
            pass
        
        return False
    
    async def _check_alumni_directory(
        self, 
        founder_name: str, 
        uni_profile: UniversityProfile, 
        degree_type: str, 
        year: str
    ) -> bool:
        """Check if founder appears in alumni directory."""
        # Most alumni directories are not publicly searchable
        # This would require specific university integrations
        return False
    
    def _create_unverified_record(
        self, 
        degree_claim: Dict[str, str], 
        founder_name: str
    ) -> EducationRecord:
        """Create unverified education record from claim."""
        return EducationRecord(
            institution=degree_claim.get("institution", ""),
            degree_type=degree_claim.get("degree", ""),
            field_of_study=degree_claim.get("field", ""),
            graduation_year=int(degree_claim["year"]) if degree_claim.get("year") else None,
            verification_status="unverified",
            verification_sources=["claimed"],
            confidence_score=0.3
        )
    
    def _deduplicate_education_records(
        self, 
        records: List[EducationRecord]
    ) -> List[EducationRecord]:
        """Remove duplicate education records."""
        seen = set()
        unique_records = []
        
        for record in records:
            key = (record.institution.lower(), record.degree_type.upper())
            if key not in seen:
                seen.add(key)
                unique_records.append(record)
        
        return unique_records
