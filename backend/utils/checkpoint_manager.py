"""Robust pickle-based checkpointing system for pipeline reliability."""

import pickle
import os
import logging
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import asyncio
import shutil

logger = logging.getLogger(__name__)


class PipelineCheckpointManager:
    """Pickle-based checkpointing for pipeline reliability and resume capability."""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self.metadata_file = self.checkpoint_dir / "metadata.json"
        self.active_jobs = {}  # Track active pipeline jobs
    
    def create_job_id(self, params: Dict[str, Any]) -> str:
        """Create unique job ID based on parameters."""
        # Create hash of parameters for consistent job IDs
        param_str = str(sorted(params.items()))
        job_hash = hashlib.md5(param_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"job_{timestamp}_{job_hash}"
    
    def save_checkpoint(self, job_id: str, stage: str, data: Any) -> bool:
        """Save checkpoint data with atomic write."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{job_id}_{stage}.pkl"
            temp_file = checkpoint_file.with_suffix('.tmp')
            
            # Atomic write: write to temp file first, then rename
            with open(temp_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now(),
                    'stage': stage,
                    'job_id': job_id
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Atomic move
            shutil.move(str(temp_file), str(checkpoint_file))
            
            logger.info(f"✅ Checkpoint saved: {job_id}_{stage}")
            self._update_job_metadata(job_id, stage, len(data) if hasattr(data, '__len__') else 1)
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to save checkpoint {job_id}_{stage}: {e}")
            # Clean up temp file if it exists
            if temp_file.exists():
                temp_file.unlink()
            return False
    
    def load_checkpoint(self, job_id: str, stage: str) -> Optional[Any]:
        """Load checkpoint data if it exists and is valid."""
        try:
            checkpoint_file = self.checkpoint_dir / f"{job_id}_{stage}.pkl"
            
            if not checkpoint_file.exists():
                return None
            
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            # Validate checkpoint
            if checkpoint['job_id'] != job_id or checkpoint['stage'] != stage:
                logger.warning(f"⚠️ Invalid checkpoint file: {checkpoint_file}")
                return None
            
            # Check if checkpoint is too old (24 hours)
            age = datetime.now() - checkpoint['timestamp']
            if age > timedelta(hours=24):
                logger.warning(f"⚠️ Checkpoint expired (age: {age}): {checkpoint_file}")
                return None
            
            logger.info(f"📂 Loaded checkpoint: {job_id}_{stage}")
            return checkpoint['data']
            
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint {job_id}_{stage}: {e}")
            return None
    
    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get progress information for a job."""
        stages = ['companies', 'enhanced_companies', 'profiles', 'founder_intelligence', 'rankings']
        progress = {
            'job_id': job_id,
            'stages': {},
            'current_stage': None,
            'completion_percentage': 0
        }
        
        completed_stages = 0
        for stage in stages:
            checkpoint_file = self.checkpoint_dir / f"{job_id}_{stage}.pkl"
            if checkpoint_file.exists():
                try:
                    with open(checkpoint_file, 'rb') as f:
                        checkpoint = pickle.load(f)
                    progress['stages'][stage] = {
                        'completed': True,
                        'timestamp': checkpoint['timestamp'],
                        'data_count': len(checkpoint['data']) if hasattr(checkpoint['data'], '__len__') else 1
                    }
                    completed_stages += 1
                    progress['current_stage'] = stage
                except Exception:
                    progress['stages'][stage] = {'completed': False}
            else:
                progress['stages'][stage] = {'completed': False}
        
        progress['completion_percentage'] = (completed_stages / len(stages)) * 100
        return progress
    
    def resume_pipeline(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Resume pipeline from the latest checkpoint."""
        progress = self.get_job_progress(job_id)
        
        if progress['completion_percentage'] == 100:
            # Job already complete, return final data
            return self.load_checkpoint(job_id, 'rankings')
        
        # Find the latest completed stage
        latest_stage = None
        latest_data = None
        
        for stage in ['rankings', 'founder_intelligence', 'profiles', 'enhanced_companies', 'companies']:
            if progress['stages'].get(stage, {}).get('completed'):
                latest_stage = stage
                latest_data = self.load_checkpoint(job_id, stage)
                break
        
        if latest_data is None:
            logger.info(f"🆕 No checkpoints found for {job_id}, starting fresh")
            return None
        
        logger.info(f"🔄 Resuming {job_id} from stage: {latest_stage}")
        return {
            'stage': latest_stage,
            'data': latest_data,
            'progress': progress
        }
    
    def cleanup_old_checkpoints(self, max_age_hours: int = 24) -> int:
        """Clean up old checkpoint files."""
        cleaned_count = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                # Check file modification time
                mod_time = datetime.fromtimestamp(checkpoint_file.stat().st_mtime)
                if mod_time < cutoff_time:
                    checkpoint_file.unlink()
                    cleaned_count += 1
                    logger.info(f"🗑️ Cleaned old checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.error(f"Failed to clean checkpoint {checkpoint_file}: {e}")
        
        return cleaned_count
    
    def list_active_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs with checkpoints."""
        jobs = {}
        
        for checkpoint_file in self.checkpoint_dir.glob("*.pkl"):
            try:
                # Extract job_id from filename
                parts = checkpoint_file.stem.split('_')
                if len(parts) >= 4:  # job_YYYYMMDD_HHMM_hash_stage
                    job_id = '_'.join(parts[:-1])  # Everything except the last part (stage)
                    
                    if job_id not in jobs:
                        jobs[job_id] = self.get_job_progress(job_id)
            except Exception:
                continue
        
        return list(jobs.values())
    
    def delete_job_checkpoints(self, job_id: str) -> int:
        """Delete all checkpoints for a specific job."""
        deleted_count = 0
        
        for checkpoint_file in self.checkpoint_dir.glob(f"{job_id}_*.pkl"):
            try:
                checkpoint_file.unlink()
                deleted_count += 1
                logger.info(f"🗑️ Deleted checkpoint: {checkpoint_file.name}")
            except Exception as e:
                logger.error(f"Failed to delete checkpoint {checkpoint_file}: {e}")
        
        return deleted_count
    
    def _update_job_metadata(self, job_id: str, stage: str, data_count: int):
        """Update job metadata for tracking."""
        self.active_jobs[job_id] = {
            'last_stage': stage,
            'last_update': datetime.now(),
            'data_count': data_count
        }
    
    def get_checkpoint_stats(self) -> Dict[str, Any]:
        """Get statistics about checkpoints."""
        checkpoint_files = list(self.checkpoint_dir.glob("*.pkl"))
        
        stats = {
            'total_checkpoints': len(checkpoint_files),
            'total_size_mb': 0,
            'active_jobs': len(self.list_active_jobs()),
            'oldest_checkpoint': None,
            'newest_checkpoint': None
        }
        
        if checkpoint_files:
            total_size = sum(f.stat().st_size for f in checkpoint_files)
            stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)
            
            mod_times = [datetime.fromtimestamp(f.stat().st_mtime) for f in checkpoint_files]
            stats['oldest_checkpoint'] = min(mod_times)
            stats['newest_checkpoint'] = max(mod_times)
        
        return stats


# Global checkpoint manager instance
checkpoint_manager = PipelineCheckpointManager()


class CheckpointedPipelineRunner:
    """Pipeline runner with integrated checkpointing."""
    
    def __init__(self, checkpoint_manager: PipelineCheckpointManager):
        self.checkpoint_manager = checkpoint_manager
    
    async def _export_companies_csv(self, companies: List, job_id: str):
        """Export companies to CSV immediately after discovery."""
        import csv
        from pathlib import Path
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate CSV filename with job ID
            csv_path = output_dir / f"{job_id}_companies.csv"
            
            if not companies:
                logger.warning("No companies to export")
                return
            
            # Define CSV columns for companies
            columns = [
                'name', 'description', 'short_description', 'founded_year',
                'funding_total_usd', 'funding_stage', 'founders', 'investors',
                'categories', 'city', 'region', 'country', 'ai_focus', 'sector',
                'website', 'linkedin_url', 'source_url', 'extraction_date'
            ]
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                
                for company in companies:
                    row = {
                        'name': getattr(company, 'name', ''),
                        'description': getattr(company, 'description', ''),
                        'short_description': getattr(company, 'short_description', ''),
                        'founded_year': getattr(company, 'founded_year', ''),
                        'funding_total_usd': getattr(company, 'funding_total_usd', ''),
                        'funding_stage': getattr(company, 'funding_stage', ''),
                        'founders': '|'.join(getattr(company, 'founders', [])),
                        'investors': '|'.join(getattr(company, 'investors', [])),
                        'categories': '|'.join(getattr(company, 'categories', [])),
                        'city': getattr(company, 'city', ''),
                        'region': getattr(company, 'region', ''),
                        'country': getattr(company, 'country', ''),
                        'ai_focus': getattr(company, 'ai_focus', ''),
                        'sector': getattr(company, 'sector', ''),
                        'website': getattr(company, 'website', ''),
                        'linkedin_url': getattr(company, 'linkedin_url', ''),
                        'source_url': getattr(company, 'source_url', ''),
                        'extraction_date': getattr(company, 'extraction_date', '')
                    }
                    writer.writerow(row)
            
            logger.info(f"💾 Exported {len(companies)} companies to {csv_path}")
            print(f"💾 Companies CSV saved: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to export companies CSV: {e}")
            print(f"❌ Failed to export companies CSV: {e}")
    
    async def _export_founders_csv(self, enriched_companies: List, job_id: str):
        """Export founders to CSV with ranking data included."""
        import csv
        from pathlib import Path
        
        try:
            # Create output directory if it doesn't exist
            output_dir = Path("./output")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate CSV filename with job ID
            csv_path = output_dir / f"{job_id}_founders.csv"
            
            # Extract all founder profiles with ranking data
            founder_records = []
            for ec in enriched_companies:
                company_name = ec.company.name
                for profile in ec.profiles:
                    founder_records.append({
                        'company_name': company_name,
                        'name': getattr(profile, 'name', ''),
                        'title': getattr(profile, 'title', ''),
                        'linkedin_url': getattr(profile, 'linkedin_url', ''),
                        'location': getattr(profile, 'location', ''),
                        'experience': getattr(profile, 'experience', ''),
                        'education': getattr(profile, 'education', ''),
                        'skills': '|'.join(getattr(profile, 'skills', [])),
                        'summary': getattr(profile, 'summary', ''),
                        'connection_count': getattr(profile, 'connection_count', ''),
                        'industry': getattr(profile, 'industry', ''),
                        'l_level': getattr(profile, 'l_level', ''),  # Ranking level
                        'confidence_score': getattr(profile, 'confidence_score', ''),  # Ranking confidence
                        'reasoning': getattr(profile, 'reasoning', ''),  # Ranking reasoning
                        'extraction_date': getattr(profile, 'extraction_date', '')
                    })
            
            if not founder_records:
                logger.warning("No founders to export")
                return
            
            # Define CSV columns for founders including ranking columns
            columns = [
                'company_name', 'name', 'title', 'linkedin_url', 'location',
                'experience', 'education', 'skills', 'summary', 'connection_count',
                'industry', 'l_level', 'confidence_score', 'reasoning', 'extraction_date'
            ]
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                writer.writerows(founder_records)
            
            logger.info(f"💾 Exported {len(founder_records)} founders to {csv_path}")
            print(f"💾 Founders CSV saved: {csv_path}")
            
        except Exception as e:
            logger.error(f"Failed to export founders CSV: {e}")
            print(f"❌ Failed to export founders CSV: {e}")
    
    
    async def resume_checkpointed_pipeline(
        self,
        checkpoint_id: str,
        pipeline_service,
        ranking_service,
        resume_data: Dict[str, Any]
    ):
        """Resume pipeline from a specific checkpoint."""
        logger.info(f"🔄 Resuming pipeline from checkpoint: {checkpoint_id}")
        
        try:
            stage = resume_data.get('stage')
            
            if stage == 'complete':
                logger.info(f"✅ Pipeline {checkpoint_id} already complete")
                return resume_data['data']
            
            # Load existing data based on completed stage
            companies = None
            enhanced_companies = None
            profiles = None
            founder_intelligence = None
            rankings = None
            
            if stage in ['companies', 'enhanced_companies', 'profiles', 'founder_intelligence', 'rankings']:
                companies = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'companies')
                
            if stage in ['enhanced_companies', 'profiles', 'founder_intelligence', 'rankings']:
                enhanced_companies = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'enhanced_companies')
                
            if stage in ['profiles', 'founder_intelligence', 'rankings']:
                profiles = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'profiles')
                
            if stage in ['founder_intelligence', 'rankings']:
                founder_intelligence = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'founder_intelligence')
                
            if stage == 'rankings':
                rankings = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'rankings')
            
            # Continue from where we left off
            if companies is None:
                logger.info("🔍 Stage 1: Company Discovery (from scratch)")
                companies = await pipeline_service.discover_companies(limit=50)
                self.checkpoint_manager.save_checkpoint(checkpoint_id, 'companies', companies)
            
            if enhanced_companies is None:
                logger.info("🔄 Stage 1.5: Company Enhancement")
                enhanced_companies = await pipeline_service.enhance_companies(companies)
                self.checkpoint_manager.save_checkpoint(checkpoint_id, 'enhanced_companies', enhanced_companies)
            
            if profiles is None:
                logger.info("👤 Stage 3: Profile Enrichment")
                profiles = await pipeline_service.enrich_profiles(enhanced_companies)
                self.checkpoint_manager.save_checkpoint(checkpoint_id, 'profiles', profiles)
            
            if founder_intelligence is None:
                logger.info("🧠 Stage 3.5: Founder Intelligence Collection")
                print("🧠 Stage 3.5: Founder Intelligence Collection - STARTING NOW")
                # Import the founder pipeline
                from ..core.data.founder_pipeline import FounderDataPipeline
                
                # Process each enriched company's LinkedIn profiles
                all_enriched_profiles = []
                print(f"📊 Found {len(profiles)} companies to process for founder intelligence")
                async with FounderDataPipeline() as founder_pipeline:
                    for i, enriched_company in enumerate(profiles):
                        if enriched_company.profiles:
                            logger.info(f"Processing {len(enriched_company.profiles)} founder profiles for {enriched_company.company.name}")
                            print(f"🔍 [{i+1}/{len(profiles)}] Processing {len(enriched_company.profiles)} founder profiles for {enriched_company.company.name}")
                            
                            try:
                                # Convert LinkedIn profiles to FounderProfiles and collect intelligence with timeout
                                company_founder_profiles = await asyncio.wait_for(
                                    founder_pipeline.collect_founder_intelligence_from_linkedin_profiles(
                                        enriched_company.profiles,
                                        enriched_company.company.name
                                    ),
                                    timeout=100  # 5 minute timeout per company
                                )
                                
                                # Update the enriched company with the new FounderProfile objects
                                enriched_company.profiles = company_founder_profiles
                                all_enriched_profiles.append(enriched_company)
                                print(f"✅ [{i+1}/{len(profiles)}] Completed intelligence collection for {enriched_company.company.name}")
                            except asyncio.TimeoutError:
                                logger.error(f"⏰ Intelligence collection timeout for {enriched_company.company.name} (5 minute limit)")
                                print(f"⏰ [{i+1}/{len(profiles)}] Intelligence collection timeout for {enriched_company.company.name} (5 minute limit)")
                                # Keep original profiles as fallback
                                all_enriched_profiles.append(enriched_company)
                            except Exception as e:
                                logger.error(f"❌ Intelligence collection failed for {enriched_company.company.name}: {e}")
                                print(f"❌ [{i+1}/{len(profiles)}] Intelligence collection failed for {enriched_company.company.name}: {e}")
                                # Keep original profiles as fallback
                                all_enriched_profiles.append(enriched_company)
                        else:
                            # Keep companies without profiles as is
                            all_enriched_profiles.append(enriched_company)
                            print(f"⚠️ [{i+1}/{len(profiles)}] No profiles found for {enriched_company.company.name}")
                
                founder_intelligence = all_enriched_profiles
                self.checkpoint_manager.save_checkpoint(checkpoint_id, 'founder_intelligence', founder_intelligence)
                logger.info(f"✅ Founder intelligence collection complete for {len(founder_intelligence)} companies")
                print(f"✅ Founder intelligence collection complete for {len(founder_intelligence)} companies")
            
            if rankings is None and ranking_service is not None:
                logger.info("🏆 Stage 4: Founder Ranking")
                # Extract founder profiles for ranking from founder intelligence data
                founder_profiles = []
                data_source = founder_intelligence if founder_intelligence else profiles
                for ec in data_source:
                    for profile in ec.profiles:
                        founder_profiles.append(profile)
                
                if founder_profiles:
                    rankings = await ranking_service.rank_founders_batch(
                        founder_profiles, 
                        batch_size=5,
                        use_enhanced=True
                    )
                    self.checkpoint_manager.save_checkpoint(checkpoint_id, 'rankings', rankings)
                else:
                    rankings = []
            
            # Mark as complete
            result = {
                'companies': founder_intelligence if founder_intelligence else profiles,
                'rankings': rankings or [],
                'job_id': checkpoint_id
            }
            
            self.checkpoint_manager.save_checkpoint(checkpoint_id, 'complete', result)
            logger.info(f"✅ Pipeline {checkpoint_id} completed successfully")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Pipeline resume failed for {checkpoint_id}: {e}")
            raise

    async def run_checkpointed_pipeline(
        self,
        pipeline_service,
        ranking_service,
        params: Dict[str, Any],
        force_restart: bool = False
    ):
        """Run pipeline with automatic checkpointing and resume capability."""
        
        # Create job ID based on parameters
        job_id = self.checkpoint_manager.create_job_id(params)
        logger.info(f"🚀 Starting checkpointed pipeline: {job_id}")
        
        try:
            # Check for existing checkpoints unless force restart
            if not force_restart:
                resume_data = self.checkpoint_manager.resume_pipeline(job_id)
                if resume_data:
                    if resume_data.get('stage') == 'complete':
                        logger.info(f"✅ Job {job_id} already complete")
                        return resume_data['data']
                    else:
                        logger.info(f"🔄 Resuming from stage: {resume_data['stage']}")
            
            # Stage 1: Company Discovery
            companies = self.checkpoint_manager.load_checkpoint(job_id, 'companies')
            if companies is None:
                logger.info("🔍 Stage 1: Company Discovery")
                companies = await pipeline_service.discover_companies(
                    limit=params.get('limit', 50),
                    categories=params.get('categories'),
                    regions=params.get('regions'),
                    founded_after=params.get('founded_after'),
                    founded_before=params.get('founded_before')
                )
                self.checkpoint_manager.save_checkpoint(job_id, 'companies', companies)
                
                # Export companies CSV immediately after discovery
                await self._export_companies_csv(companies, job_id)
                
            else:
                logger.info("📂 Stage 1: Loaded companies from checkpoint")
                # Check if companies CSV already exists, if not export it
                from pathlib import Path
                csv_path = Path("./output") / f"{job_id}_companies.csv"
                if not csv_path.exists():
                    await self._export_companies_csv(companies, job_id)
            
            # Stage 1.5: Company Enhancement with Crunchbase data fusion
            enhanced_companies = self.checkpoint_manager.load_checkpoint(job_id, 'enhanced_companies')
            if enhanced_companies is None:
                logger.info("🔄 Stage 1.5: Company Enhancement")
                enhanced_companies = await pipeline_service.enhance_companies(companies)
                self.checkpoint_manager.save_checkpoint(job_id, 'enhanced_companies', enhanced_companies)
            else:
                logger.info("📂 Stage 1.5: Loaded enhanced companies from checkpoint")
            
            # Stage 3: Profile Enrichment
            profiles = self.checkpoint_manager.load_checkpoint(job_id, 'profiles')
            if profiles is None:
                logger.info("👤 Stage 3: Profile Enrichment")
                profiles = await pipeline_service.enrich_profiles(enhanced_companies)
                self.checkpoint_manager.save_checkpoint(job_id, 'profiles', profiles)
                
            else:
                logger.info("📂 Stage 3: Loaded profiles from checkpoint")
            
            # Stage 3.5: Founder Intelligence Collection
            founder_intelligence = self.checkpoint_manager.load_checkpoint(job_id, 'founder_intelligence')
            if founder_intelligence is None:
                logger.info("🧠 Stage 3.5: Founder Intelligence Collection")
                # Import the founder pipeline
                from ..core.data.founder_pipeline import FounderDataPipeline
                
                # Process each enriched company's LinkedIn profiles
                all_enriched_profiles = []
                async with FounderDataPipeline() as founder_pipeline:
                    for enriched_company in profiles:
                        if enriched_company.profiles:
                            logger.info(f"Processing {len(enriched_company.profiles)} founder profiles for {enriched_company.company.name}")
                            
                            # Convert LinkedIn profiles to FounderProfiles and collect intelligence
                            company_founder_profiles = await founder_pipeline.collect_founder_intelligence_from_linkedin_profiles(
                                enriched_company.profiles,
                                enriched_company.company.name
                            )
                            
                            # Update the enriched company with the new FounderProfile objects
                            enriched_company.profiles = company_founder_profiles
                            all_enriched_profiles.append(enriched_company)
                        else:
                            # Keep companies without profiles as is
                            all_enriched_profiles.append(enriched_company)
                
                founder_intelligence = all_enriched_profiles
                self.checkpoint_manager.save_checkpoint(job_id, 'founder_intelligence', founder_intelligence)
                logger.info(f"✅ Founder intelligence collection complete for {len(founder_intelligence)} companies")
            else:
                logger.info("📂 Stage 3.5: Loaded founder intelligence from checkpoint")
            
            if not companies:
                raise ValueError("No companies found in discovery stage")
            
            if not founder_intelligence:
                raise ValueError("No founder intelligence found in enrichment stage")
            
            # Extract profiles data for further processing
            profiles_data = []
            for ec in founder_intelligence:
                for profile in ec.profiles:
                    profiles_data.append({
                        'company_name': ec.company.name,
                        'profile': profile
                    })
            
            logger.info(f"📂 Stage 3.5: Extracted {len(profiles_data)} enriched founder profiles")
            
            # Stage 4: Founder Rankings
            rankings = self.checkpoint_manager.load_checkpoint(job_id, 'rankings')
            if rankings is None and profiles_data:
                logger.info("🏆 Stage 4: Founder Rankings")
                
                # Profiles are already FounderProfile objects from founder intelligence stage
                founder_profiles = [pd['profile'] for pd in profiles_data]
                
                # Rank founders with enhanced system
                from ..core.ranking.ranking_service import FounderRankingService
                ranking_service = FounderRankingService()
                
                rankings = await ranking_service.rank_founders_batch(
                    founder_profiles, 
                    batch_size=5,
                    use_enhanced=True  # Use enhanced ranking with L-level validation
                )
                
                self.checkpoint_manager.save_checkpoint(job_id, 'rankings', rankings)
                
                # Export founders CSV immediately after ranking
                await self._export_founders_csv(founder_intelligence, job_id)
                
            else:
                logger.info("📂 Stage 3: Loaded rankings from checkpoint")
                # Check if founders CSV already exists, if not export it
                from pathlib import Path
                csv_path = Path("./output") / f"{job_id}_founders.csv"
                if not csv_path.exists():
                    await self._export_founders_csv(founder_intelligence, job_id)
            
            # Stage 4: Complete - Final result
            final_result = {
                'job_id': job_id,
                'companies': founder_intelligence,
                'profiles': profiles_data,
                'rankings': rankings,
                'completed_at': datetime.now(),
                'stats': {
                    'total_companies': len(founder_intelligence),
                    'total_founders': len(profiles_data),
                    'ranked_founders': len(rankings) if rankings else 0,
                    'high_confidence_founders': len([r for r in rankings if r.classification.confidence_score >= 0.75]) if rankings else 0
                }
            }
            
            self.checkpoint_manager.save_checkpoint(job_id, 'complete', final_result)
            logger.info(f"✅ Pipeline complete: {job_id}")
            
            return final_result
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {job_id} - {e}")
            raise


# Global checkpointed runner instance
checkpointed_runner = CheckpointedPipelineRunner(checkpoint_manager)
