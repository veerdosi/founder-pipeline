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
                companies = await pipeline_service.run_complete_pipeline_with_date_range(
                    company_limit=params.get('limit', 50),
                    categories=params.get('categories'),
                    regions=params.get('regions'),
                    sources=params.get('sources'),
                    founded_after=params.get('founded_after'),
                    founded_before=params.get('founded_before')
                )
                self.checkpoint_manager.save_checkpoint(job_id, 'companies', companies)
            else:
                logger.info("📂 Stage 1: Loaded companies from checkpoint")
            
            if not companies:
                raise ValueError("No companies found in discovery stage")
            
            # Stage 2: Profile Collection (already included in pipeline)
            # Companies already have profiles from pipeline
            profiles_data = []
            for ec in companies:
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
                        title=profile.title or "",
                        linkedin_url=str(profile.linkedin_url),
                        location=profile.location,
                        about=profile.about,
                        estimated_age=profile.estimated_age,
                        experience_1_title=profile.experience_1_title,
                        experience_1_company=profile.experience_1_company,
                        experience_2_title=profile.experience_2_title,
                        experience_2_company=profile.experience_2_company,
                        experience_3_title=profile.experience_3_title,
                        experience_3_company=profile.experience_3_company,
                        education_1_school=profile.education_1_school,
                        education_1_degree=profile.education_1_degree,
                        education_2_school=profile.education_2_school,
                        education_2_degree=profile.education_2_degree,
                        skill_1=profile.skill_1,
                        skill_2=profile.skill_2,
                        skill_3=profile.skill_3
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
            else:
                logger.info("📂 Stage 3: Loaded rankings from checkpoint")
            
            # Stage 4: Complete - Final result
            final_result = {
                'job_id': job_id,
                'companies': companies,
                'profiles': profiles_data,
                'rankings': rankings,
                'completed_at': datetime.now(),
                'stats': {
                    'total_companies': len(companies),
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
