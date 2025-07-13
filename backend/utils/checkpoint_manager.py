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
            
            # Validate checkpoint structure
            if not isinstance(checkpoint, dict):
                logger.warning(f"⚠️ Invalid checkpoint structure: {checkpoint_file}")
                return None
                
            # Validate checkpoint metadata
            if checkpoint.get('job_id') != job_id or checkpoint.get('stage') != stage:
                logger.warning(f"⚠️ Invalid checkpoint metadata: {checkpoint_file} (expected job_id: {job_id}, stage: {stage})")
                return None
            
            # Check if checkpoint is too old (72 hours)
            checkpoint_timestamp = checkpoint.get('timestamp')
            if checkpoint_timestamp:
                age = datetime.now() - checkpoint_timestamp
                if age > timedelta(hours=72):
                    logger.warning(f"⚠️ Checkpoint expired (age: {age}): {checkpoint_file}")
                    return None
            
            return checkpoint.get('data')
            
        except (pickle.UnpicklingError, ModuleNotFoundError, ImportError) as e:
            logger.warning(f"⚠️ Could not load checkpoint {checkpoint_file} due to missing dependencies or corrupted data: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint {job_id}_{stage}: {e}")
            return None
    
    def get_job_progress(self, job_id: str) -> Dict[str, Any]:
        """Get progress information for a job."""
        stages = ['companies', 'enhanced_companies', 'profiles', 'rankings']
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
        
        for stage in ['rankings', 'profiles', 'enhanced_companies', 'companies']:
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
                # Expected format: job_YYYYMMDD_HHMM_hash_stage
                # Stage can be: companies, enhanced_companies, profiles, rankings
                filename = checkpoint_file.stem
                
                # Find the stage suffix and extract job_id accordingly
                known_stages = ['rankings', 'profiles', 'enhanced_companies', 'companies']
                job_id = None
                
                for stage in known_stages:
                    if filename.endswith(f'_{stage}'):
                        job_id = filename[:-len(f'_{stage}')]
                        break
                
                if not job_id:
                    continue  # Skip files that don't match expected pattern
                
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
                    # Extract separate experience, education, and skills columns
                    record = {
                        'company_name': company_name,
                        'name': getattr(profile, 'person_name', ''),
                        'title': getattr(profile, 'title', ''),
                        'linkedin_url': getattr(profile, 'linkedin_url', ''),
                        'location': getattr(profile, 'location', ''),
                        'about': getattr(profile, 'about', ''),
                        'estimated_age': getattr(profile, 'estimated_age', ''),
                        'l_level': getattr(profile, 'l_level', ''),  # Ranking level
                        'confidence_score': getattr(profile, 'confidence_score', ''),  # Ranking confidence
                        'reasoning': getattr(profile, 'reasoning', ''),  # Ranking reasoning
                        'extraction_date': getattr(profile, 'extraction_date', '')
                    }
                    
                    # Add separate experience columns
                    experiences = getattr(profile, 'experience', []) or []
                    for i in range(3):  # Support up to 3 experiences
                        if i < len(experiences) and experiences[i]:
                            record[f'experience_{i+1}_title'] = experiences[i].get('title', '')
                            record[f'experience_{i+1}_company'] = experiences[i].get('company', '')
                        else:
                            record[f'experience_{i+1}_title'] = ''
                            record[f'experience_{i+1}_company'] = ''
                    
                    # Add separate education columns
                    educations = getattr(profile, 'education', []) or []
                    for i in range(2):  # Support up to 2 educations
                        if i < len(educations) and educations[i]:
                            record[f'education_{i+1}_school'] = educations[i].get('school', '')
                            record[f'education_{i+1}_degree'] = educations[i].get('degree', '')
                        else:
                            record[f'education_{i+1}_school'] = ''
                            record[f'education_{i+1}_degree'] = ''
                    
                    # Add separate skills columns
                    skills = getattr(profile, 'skills', []) or []
                    for i in range(5):  # Support up to 5 skills
                        if i < len(skills):
                            record[f'skill_{i+1}'] = skills[i]
                        else:
                            record[f'skill_{i+1}'] = ''
                    
                    founder_records.append(record)
            
            if not founder_records:
                logger.warning("No founders to export")
                return
            
            # Define CSV columns for founders including ranking columns
            columns = [
                'company_name', 'name', 'title', 'linkedin_url', 'location', 'about',
                'estimated_age', 'extraction_date',
                'experience_1_title', 'experience_1_company', 'experience_2_title', 'experience_2_company',
                'experience_3_title', 'experience_3_company', 'education_1_school', 'education_1_degree',
                'education_2_school', 'education_2_degree', 'skill_1', 'skill_2', 'skill_3', 'skill_4', 'skill_5',
                'l_level', 'reasoning', 'confidence_score'
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
            result = await pipeline_service.run(force_restart=False)
            
            final_result = {
                'job_id': checkpoint_id,
                'companies': result,
                'rankings': [],
                'completed_at': datetime.now(),
                'stats': {
                    'total_companies': len(result),
                    'total_founders': sum(len(ec.profiles) for ec in result),
                    'ranked_founders': 0,
                    'high_confidence_founders': 0
                }
            }
            self.checkpoint_manager.save_checkpoint(checkpoint_id, 'complete', final_result)
            logger.info(f"✅ Pipeline {checkpoint_id} completed successfully")
            
            return final_result
            
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
        
        job_id = self.checkpoint_manager.create_job_id(params)
        pipeline_service.job_id = job_id
        logger.info(f"🚀 Starting checkpointed pipeline: {job_id}")
        
        try:
            result = await pipeline_service.run(
                limit=params.get('limit', 50),
                categories=params.get('categories'),
                regions=params.get('regions'),
                founded_after=params.get('founded_after'),
                founded_before=params.get('founded_before'),
                force_restart=force_restart
            )

            final_result = {
                'job_id': job_id,
                'companies': result,
                'rankings': [],
                'completed_at': datetime.now(),
                'stats': {
                    'total_companies': len(result),
                    'total_founders': sum(len(ec.profiles) for ec in result),
                    'ranked_founders': 0,
                    'high_confidence_founders': 0
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
