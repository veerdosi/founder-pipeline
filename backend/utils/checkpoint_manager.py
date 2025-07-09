"""Robust pickle-based checkpointing system for pipeline reliability."""

import pickle
import os
import logging
import hashlib
from pathlib import Path
from typing import Any, Optional, Dict, List
from datetime import datetime, timedelta
import tempfile
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
        stages = ['companies', 'profiles', 'rankings', 'complete']
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
            return self.load_checkpoint(job_id, 'complete')
        
        # Find the latest completed stage
        latest_stage = None
        latest_data = None
        
        for stage in ['rankings', 'profiles', 'companies']:
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
            enriched_companies = None
            rankings = None
            
            if stage in ['companies', 'enhanced_companies', 'enriched_companies', 'rankings']:
                companies = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'companies')
                
            if stage in ['enhanced_companies', 'enriched_companies', 'rankings']:
                enhanced_companies = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'enhanced_companies')
                
            if stage in ['enriched_companies', 'rankings']:
                enriched_companies = self.checkpoint_manager.load_checkpoint(checkpoint_id, 'enriched_companies')
                
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
            
            if enriched_companies is None:
                logger.info("👤 Stage 2: Profile Enrichment")
                enriched_companies = await pipeline_service.enrich_profiles(enhanced_companies)
                self.checkpoint_manager.save_checkpoint(checkpoint_id, 'enriched_companies', enriched_companies)
            
            if rankings is None and ranking_service is not None:
                logger.info("🏆 Stage 3: Founder Ranking")
                # Extract founder profiles for ranking
                founder_profiles = []
                for ec in enriched_companies:
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
                'companies': enriched_companies,
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
            
            # Stage 2: Profile Enrichment
            enriched_companies = self.checkpoint_manager.load_checkpoint(job_id, 'enriched_companies')
            if enriched_companies is None:
                logger.info("👤 Stage 2: Profile Enrichment")
                enriched_companies = await pipeline_service.enrich_profiles(enhanced_companies)
                self.checkpoint_manager.save_checkpoint(job_id, 'enriched_companies', enriched_companies)
                
            else:
                logger.info("📂 Stage 2: Loaded enriched companies from checkpoint")
            
            if not companies:
                raise ValueError("No companies found in discovery stage")
            
            if not enriched_companies:
                raise ValueError("No enriched companies found in enrichment stage")
            
            # Extract profiles data for further processing
            profiles_data = []
            for ec in enriched_companies:
                for profile in ec.profiles:
                    profiles_data.append({
                        'company_name': ec.company.name,
                        'profile': profile
                    })
            
            self.checkpoint_manager.save_checkpoint(job_id, 'profiles', profiles_data)
            logger.info(f"📂 Stage 2: Saved {len(profiles_data)} profiles")
            
            # Stage 3: Founder Rankings
            rankings = self.checkpoint_manager.load_checkpoint(job_id, 'rankings')
            if rankings is None and profiles_data:
                logger.info("🏆 Stage 3: Founder Rankings")
                
                # Convert to FounderProfile format for ranking
                founder_profiles = []
                for pd in profiles_data:
                    profile = pd['profile']
                    from ..core.ranking.models import FounderProfile
                    founder_profile = FounderProfile(
                        name=profile.person_name,
                        company_name=pd['company_name'],
                        title=getattr(profile, 'current_position', '') or "",
                        linkedin_url=str(profile.linkedin_url) if profile.linkedin_url else "",
                        location=getattr(profile, 'location', '') or "",
                        about=getattr(profile, 'summary', '') or "",
                        estimated_age=getattr(profile, 'estimated_age', None),
                        experience_1_title=getattr(profile, 'experience_1_title', None),
                        experience_1_company=getattr(profile, 'experience_1_company', None),
                        experience_2_title=getattr(profile, 'experience_2_title', None),
                        experience_2_company=getattr(profile, 'experience_2_company', None),
                        experience_3_title=getattr(profile, 'experience_3_title', None),
                        experience_3_company=getattr(profile, 'experience_3_company', None),
                        education_1_school=getattr(profile, 'education_1_school', None),
                        education_1_degree=getattr(profile, 'education_1_degree', None),
                        education_2_school=getattr(profile, 'education_2_school', None),
                        education_2_degree=getattr(profile, 'education_2_degree', None),
                        skill_1=getattr(profile, 'skill_1', None),
                        skill_2=getattr(profile, 'skill_2', None),
                        skill_3=getattr(profile, 'skill_3', None)
                    )
                    founder_profiles.append((founder_profile, profile))
                
                # Rank founders with enhanced system
                from ..core.ranking.ranking_service import FounderRankingService
                ranking_service = FounderRankingService()
                
                rankings = await ranking_service.rank_founders_batch(
                    [fp for fp, _ in founder_profiles], 
                    batch_size=5,
                    use_enhanced=True  # Use enhanced ranking with L-level validation
                )
                
                # Apply rankings back to original profiles
                for i, ranking in enumerate(rankings):
                    if i < len(founder_profiles):
                        _, original_profile = founder_profiles[i]
                        original_profile.l_level = ranking.classification.level.value
                        original_profile.confidence_score = ranking.classification.confidence_score
                        original_profile.reasoning = ranking.classification.reasoning
                
                self.checkpoint_manager.save_checkpoint(job_id, 'rankings', rankings)
                
                # Export founders CSV immediately after ranking
                await self._export_founders_csv(enriched_companies, job_id)
                
            else:
                logger.info("📂 Stage 3: Loaded rankings from checkpoint")
                # Check if founders CSV already exists, if not export it
                from pathlib import Path
                csv_path = Path("./output") / f"{job_id}_founders.csv"
                if not csv_path.exists():
                    await self._export_founders_csv(enriched_companies, job_id)
            
            # Stage 4: Complete - Final result
            final_result = {
                'job_id': job_id,
                'companies': enriched_companies,
                'profiles': profiles_data,
                'rankings': rankings,
                'completed_at': datetime.now(),
                'stats': {
                    'total_companies': len(enriched_companies),
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
