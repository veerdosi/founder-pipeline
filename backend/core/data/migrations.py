"""Database migrations and initialization for company tracking system."""

import sqlite3
import logging
from pathlib import Path
from typing import Optional

from ..config import settings

logger = logging.getLogger(__name__)


class DatabaseMigrations:
    """Handle database migrations and initialization."""
    
    def __init__(self, db_path: Optional[Path] = None):
        """Initialize migrations with database path."""
        if db_path is None:
            if settings.company_tracking_db_path:
                db_path = settings.company_tracking_db_path
            else:
                db_path = settings.default_output_dir / "company_tracking.db"
        
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
    def run_migrations(self):
        """Run all database migrations in order."""
        logger.info(f"Running database migrations for: {self.db_path}")
        
        try:
            self._create_initial_schema()
            self._add_indexes()
            self._add_tracking_enhancements()
            logger.info("Database migrations completed successfully")
        except Exception as e:
            logger.error(f"Database migration failed: {e}")
            raise
    
    def _create_initial_schema(self):
        """Create the initial database schema."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS companies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    uuid TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    normalized_name TEXT NOT NULL,
                    website TEXT,
                    website_domain TEXT,
                    description TEXT,
                    description_hash TEXT,
                    discovered_date TEXT NOT NULL,
                    last_seen TEXT NOT NULL,
                    founded_year INTEGER,
                    city TEXT,
                    region TEXT,
                    country TEXT,
                    status TEXT DEFAULT 'active',
                    discovery_count INTEGER DEFAULT 1,
                    source_urls TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    
                    -- Ensure uniqueness
                    UNIQUE(normalized_name, website_domain)
                );
                
                CREATE TABLE IF NOT EXISTS discovery_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT UNIQUE NOT NULL,
                    run_date TEXT NOT NULL,
                    job_id TEXT,
                    target_year INTEGER,
                    companies_found INTEGER DEFAULT 0,
                    companies_new INTEGER DEFAULT 0,
                    companies_duplicate INTEGER DEFAULT 0,
                    execution_time_seconds REAL,
                    status TEXT DEFAULT 'running',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    completed_at TEXT
                );
                
                -- Version tracking table
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY,
                    version INTEGER NOT NULL,
                    applied_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                -- Insert initial version if not exists
                INSERT OR IGNORE INTO schema_version (id, version) VALUES (1, 1);
            """)
            logger.info("Initial schema created")
    
    def _add_indexes(self):
        """Add database indexes for performance."""
        with sqlite3.connect(str(self.db_path)) as conn:
            conn.executescript("""
                CREATE INDEX IF NOT EXISTS idx_companies_normalized_name ON companies(normalized_name);
                CREATE INDEX IF NOT EXISTS idx_companies_website_domain ON companies(website_domain);
                CREATE INDEX IF NOT EXISTS idx_companies_discovered_date ON companies(discovered_date);
                CREATE INDEX IF NOT EXISTS idx_companies_status ON companies(status);
                CREATE INDEX IF NOT EXISTS idx_companies_founded_year ON companies(founded_year);
                CREATE INDEX IF NOT EXISTS idx_discovery_runs_date ON discovery_runs(run_date);
                CREATE INDEX IF NOT EXISTS idx_discovery_runs_target_year ON discovery_runs(target_year);
                CREATE INDEX IF NOT EXISTS idx_discovery_runs_status ON discovery_runs(status);
            """)
            logger.info("Database indexes created")
    
    def _add_tracking_enhancements(self):
        """Add enhanced tracking features."""
        with sqlite3.connect(str(self.db_path)) as conn:
            cursor = conn.cursor()
            
            # Check current schema version
            cursor.execute("SELECT MAX(version) FROM schema_version")
            current_version = cursor.fetchone()[0] or 1
            
            if current_version < 2:
                # Add additional columns if they don't exist
                try:
                    cursor.execute("ALTER TABLE companies ADD COLUMN linkedin_url TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    cursor.execute("ALTER TABLE companies ADD COLUMN crunchbase_url TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    cursor.execute("ALTER TABLE companies ADD COLUMN funding_total_usd REAL")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                try:
                    cursor.execute("ALTER TABLE companies ADD COLUMN funding_stage TEXT")
                except sqlite3.OperationalError:
                    pass  # Column already exists
                
                # Update schema version
                cursor.execute("INSERT INTO schema_version (version) VALUES (2)")
                conn.commit()
                logger.info("Enhanced tracking features added (version 2)")
    
    def get_database_info(self) -> dict:
        """Get information about the database."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                
                # Get table info
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
                """)
                tables = [row['name'] for row in cursor.fetchall()]
                
                # Get schema version
                cursor.execute("SELECT MAX(version) FROM schema_version")
                version = cursor.fetchone()[0] or 1
                
                # Get record counts
                record_counts = {}
                for table in tables:
                    cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
                    record_counts[table] = cursor.fetchone()['count']
                
                return {
                    'database_path': str(self.db_path),
                    'database_exists': self.db_path.exists(),
                    'database_size_mb': self.db_path.stat().st_size / 1024 / 1024 if self.db_path.exists() else 0,
                    'schema_version': version,
                    'tables': tables,
                    'record_counts': record_counts
                }
                
        except Exception as e:
            logger.error(f"Error getting database info: {e}")
            return {'error': str(e)}
    
    def reset_database(self):
        """Reset the entire database (DANGER: This will delete all data)."""
        if self.db_path.exists():
            self.db_path.unlink()
            logger.warning(f"Database reset - all data deleted: {self.db_path}")
        
        self.run_migrations()
        logger.info("Database reset and reinitialized")
    
    def vacuum_database(self):
        """Optimize database by running VACUUM."""
        try:
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuum completed")
        except Exception as e:
            logger.error(f"Database vacuum failed: {e}")


def initialize_database(db_path: Optional[Path] = None) -> DatabaseMigrations:
    """Initialize the database with migrations."""
    migrations = DatabaseMigrations(db_path)
    migrations.run_migrations()
    return migrations


def get_database_info(db_path: Optional[Path] = None) -> dict:
    """Get database information."""
    migrations = DatabaseMigrations(db_path)
    return migrations.get_database_info()


# Run migrations on import if tracking is enabled
if settings.company_tracking_enabled:
    try:
        initialize_database()
        logger.info("Company tracking database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize company tracking database: {e}")