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
            
            # Check if checkpoint is too old (15 days)
            checkpoint_timestamp = checkpoint.get('timestamp')
            if checkpoint_timestamp:
                age = datetime.now() - checkpoint_timestamp
                if age > timedelta(hours=360):
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
    
    async def _export_companies_csv(self, companies: List, job_id: str, target_year: Optional[int] = None):
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
                'name', 'description', 'founded_year',
                'funding_total_usd', 'funding_stage', 'founders', 'investors',
                'categories', 'city', 'region', 'country', 'sector',
                'website', 'linkedin_url', 'crunchbase_url', 'source_url', 'extraction_date',
                # Market analysis metrics
                'market_size_billion', 'cagr_percent', 'timing_score', 'competitor_count',
                'market_stage', 'confidence_score_market', 'us_sentiment', 'sea_sentiment',
                'total_funding_billion', 'momentum_score'
            ]
            
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=columns)
                writer.writeheader()
                
                for company in companies:
                    # Handle both Company and EnrichedCompany objects
                    if hasattr(company, 'company'):
                        # This is an EnrichedCompany, access the nested company
                        comp = company.company
                    else:
                        # This is a regular Company object
                        comp = company
                    
                    # Use target_year as fallback if founded_year is None
                    founded_year = getattr(comp, 'founded_year', None)
                    if founded_year is None and target_year is not None:
                        founded_year = target_year
                    
                    # Extract market metrics if available
                    market_metrics = getattr(comp, 'market_metrics', None)
                    
                    row = {
                        'name': getattr(comp, 'name', ''),
                        'description': getattr(comp, 'description', ''),
                        'founded_year': founded_year if founded_year is not None else '',
                        'funding_total_usd': getattr(comp, 'funding_total_usd', ''),
                        'funding_stage': getattr(comp, 'funding_stage', ''),
                        'founders': '|'.join(getattr(comp, 'founders', [])),
                        'investors': '|'.join(getattr(comp, 'investors', [])),
                        'categories': '|'.join(getattr(comp, 'categories', [])),
                        'city': getattr(comp, 'city', ''),
                        'region': getattr(comp, 'region', ''),
                        'country': getattr(comp, 'country', ''),
                        'sector': getattr(comp, 'sector', ''),
                        'website': getattr(comp, 'website', ''),
                        'linkedin_url': getattr(comp, 'linkedin_url', ''),
                        'crunchbase_url': getattr(comp, 'crunchbase_url', ''),
                        'source_url': getattr(comp, 'source_url', ''),
                        'extraction_date': getattr(comp, 'extraction_date', ''),
                        # Market analysis metrics
                        'market_size_billion': getattr(market_metrics, 'market_size_billion', '') if market_metrics else '',
                        'cagr_percent': getattr(market_metrics, 'cagr_percent', '') if market_metrics else '',
                        'timing_score': getattr(market_metrics, 'timing_score', '') if market_metrics else '',
                        'competitor_count': getattr(market_metrics, 'competitor_count', '') if market_metrics else '',
                        'market_stage': getattr(market_metrics, 'market_stage', '') if market_metrics else '',
                        'confidence_score_market': getattr(market_metrics, 'confidence_score', '') if market_metrics else '',
                        'us_sentiment': getattr(market_metrics, 'us_sentiment', '') if market_metrics else '',
                        'sea_sentiment': getattr(market_metrics, 'sea_sentiment', '') if market_metrics else '',
                        'total_funding_billion': getattr(market_metrics, 'total_funding_billion', '') if market_metrics else '',
                        'momentum_score': getattr(market_metrics, 'momentum_score', '') if market_metrics else ''
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
                    # Format confidence score to 2 decimal places
                    confidence_score = getattr(profile, 'confidence_score', '')
                    if confidence_score and isinstance(confidence_score, (int, float)):
                        confidence_score = f"{confidence_score:.2f}"
                    
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
                        'confidence_score': confidence_score,  # Ranking confidence (formatted to 2 decimal places)
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
                    
                    # Add media coverage data
                    media_coverage = getattr(profile, 'media_coverage', None)
                    if media_coverage:
                        record['media_mentions_count'] = getattr(media_coverage, 'media_mentions_count', '')
                        record['awards_and_recognitions'] = '; '.join(getattr(media_coverage, 'awards_and_recognitions', []) or [])
                        record['speaking_engagements'] = '; '.join(getattr(media_coverage, 'speaking_engagements', []) or [])
                        record['social_media_followers'] = getattr(media_coverage, 'social_media_followers', '')
                        record['thought_leadership_score'] = getattr(media_coverage, 'thought_leadership_score', '')
                        record['overall_sentiment'] = getattr(media_coverage, 'overall_sentiment', '')
                    else:
                        record['media_mentions_count'] = ''
                        record['awards_and_recognitions'] = ''
                        record['speaking_engagements'] = ''
                        record['social_media_followers'] = ''
                        record['thought_leadership_score'] = ''
                        record['overall_sentiment'] = ''
                    
                    # Add financial profile data
                    financial_profile = getattr(profile, 'financial_profile', None)
                    if financial_profile:
                        # Flatten companies founded
                        companies_founded = getattr(financial_profile, 'companies_founded', []) or []
                        record['companies_founded'] = '; '.join([
                            f"{c.get('name', '')} ({c.get('founding_year', '')})" 
                            for c in companies_founded if isinstance(c, dict)
                        ])
                        
                        # Flatten investment activities
                        investments = getattr(financial_profile, 'investment_activities', []) or []
                        record['investment_activities'] = '; '.join([
                            f"{inv.get('company', '')} - ${inv.get('amount', '')}" 
                            for inv in investments if isinstance(inv, dict)
                        ])
                        
                        # Flatten board positions
                        board_positions = getattr(financial_profile, 'board_positions', []) or []
                        record['board_positions'] = '; '.join([
                            f"{pos.get('company', '')} ({pos.get('position', '')})" 
                            for pos in board_positions if isinstance(pos, dict)
                        ])
                        
                        record['notable_achievements'] = '; '.join(getattr(financial_profile, 'notable_achievements', []) or [])
                        record['estimated_net_worth'] = getattr(financial_profile, 'estimated_net_worth', '')
                        record['confidence_level'] = getattr(financial_profile, 'confidence_level', '')
                    else:
                        record['companies_founded'] = ''
                        record['investment_activities'] = ''
                        record['board_positions'] = ''
                        record['notable_achievements'] = ''
                        record['estimated_net_worth'] = ''
                        record['confidence_level'] = ''
                    
                    founder_records.append(record)
            
            if not founder_records:
                logger.warning("No founders to export")
                return
            
            # Define CSV columns for founders including ranking and enhancement columns
            columns = [
                'company_name', 'name', 'title', 'linkedin_url', 'location', 'about',
                'estimated_age', 'extraction_date',
                'experience_1_title', 'experience_1_company', 'experience_2_title', 'experience_2_company',
                'experience_3_title', 'experience_3_company', 'education_1_school', 'education_1_degree',
                'education_2_school', 'education_2_degree', 'skill_1', 'skill_2', 'skill_3', 'skill_4', 'skill_5',
                'media_mentions_count', 'awards_and_recognitions', 'speaking_engagements', 'social_media_followers',
                'thought_leadership_score', 'overall_sentiment', 'companies_founded', 'investment_activities',
                'board_positions', 'notable_achievements', 'estimated_net_worth', 'confidence_level',
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
                limit=params.get('limit', settings.default_company_limit),
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
