"""File handling utilities for the pipeline."""

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Dict, List, Union

import aiofiles
import pandas as pd
from rich.console import Console

console = Console()


async def read_json_async(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Read JSON file asynchronously."""
    async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
        content = await f.read()
        return json.loads(content)


async def write_json_async(data: Any, file_path: Union[str, Path]) -> None:
    """Write data to JSON file asynchronously."""
    async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(data, indent=2, default=str, ensure_ascii=False))


def read_csv_safe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read CSV file with error handling."""
    try:
        return pd.read_csv(file_path, **kwargs)
    except Exception as e:
        console.print(f"❌ Error reading CSV {file_path}: {e}")
        return pd.DataFrame()


def write_csv_safe(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """Write DataFrame to CSV with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(file_path, index=False, **kwargs)
        return True
    except Exception as e:
        console.print(f"❌ Error writing CSV {file_path}: {e}")
        return False


def read_excel_safe(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Read Excel file with error handling."""
    try:
        return pd.read_excel(file_path, **kwargs)
    except Exception as e:
        console.print(f"❌ Error reading Excel {file_path}: {e}")
        return pd.DataFrame()


def write_excel_safe(df: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> bool:
    """Write DataFrame to Excel with error handling."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(file_path, index=False, **kwargs)
        return True
    except Exception as e:
        console.print(f"❌ Error writing Excel {file_path}: {e}")
        return False


async def save_checkpoint_async(data: Any, file_path: Union[str, Path]) -> bool:
    """Save checkpoint data asynchronously."""
    try:
        # Ensure directory exists
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(pickle.dumps(data))
        
        console.print(f"✅ Checkpoint saved: {file_path}")
        return True
    except Exception as e:
        console.print(f"❌ Error saving checkpoint {file_path}: {e}")
        return False


async def load_checkpoint_async(file_path: Union[str, Path]) -> Any:
    """Load checkpoint data asynchronously."""
    try:
        if not Path(file_path).exists():
            return None
            
        async with aiofiles.open(file_path, 'rb') as f:
            data = await f.read()
        
        console.print(f"✅ Checkpoint loaded: {file_path}")
        return pickle.loads(data)
    except Exception as e:
        console.print(f"❌ Error loading checkpoint {file_path}: {e}")
        return None


def get_file_size(file_path: Union[str, Path]) -> int:
    """Get file size in bytes."""
    try:
        return Path(file_path).stat().st_size
    except:
        return 0


def ensure_directory(file_path: Union[str, Path]) -> Path:
    """Ensure directory exists for file path."""
    path = Path(file_path)
    if path.suffix:  # It's a file path
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.parent
    else:  # It's a directory path
        path.mkdir(parents=True, exist_ok=True)
        return path


def list_files_by_extension(directory: Union[str, Path], extension: str) -> List[Path]:
    """List all files with specific extension in directory."""
    directory = Path(directory)
    if not directory.exists():
        return []
    
    pattern = f"*.{extension.lstrip('.')}"
    return list(directory.glob(pattern))


def backup_file(file_path: Union[str, Path], backup_suffix: str = ".bak") -> bool:
    """Create a backup of a file."""
    try:
        source = Path(file_path)
        if not source.exists():
            return False
        
        backup_path = source.with_suffix(source.suffix + backup_suffix)
        backup_path.write_bytes(source.read_bytes())
        
        console.print(f"✅ Backup created: {backup_path}")
        return True
    except Exception as e:
        console.print(f"❌ Error creating backup: {e}")
        return False


def clean_filename(filename: str) -> str:
    """Clean filename to be filesystem-safe."""
    import re
    
    # Remove or replace invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove multiple consecutive underscores
    filename = re.sub(r'_{2,}', '_', filename)
    
    # Remove leading/trailing underscores and dots
    filename = filename.strip('_.')
    
    # Limit length
    if len(filename) > 200:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:200-len(ext)-1] + ('.' + ext if ext else '')
    
    return filename


class FileManager:
    """File management utilities."""
    
    def __init__(self, base_dir: Union[str, Path]):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def get_path(self, *parts: str) -> Path:
        """Get path relative to base directory."""
        return self.base_dir / Path(*parts)
    
    def save_data(self, data: Any, filename: str, format: str = "json") -> Path:
        """Save data in specified format."""
        file_path = self.get_path(filename)
        
        if format.lower() == "json":
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, default=str, ensure_ascii=False)
        elif format.lower() == "csv" and isinstance(data, (list, dict)):
            if isinstance(data, list) and data and isinstance(data[0], dict):
                df = pd.DataFrame(data)
                df.to_csv(file_path, index=False)
            else:
                raise ValueError("Data must be list of dictionaries for CSV format")
        elif format.lower() == "pickle":
            with open(file_path, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return file_path
    
    def load_data(self, filename: str, format: str = "json") -> Any:
        """Load data from file."""
        file_path = self.get_path(filename)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if format.lower() == "json":
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        elif format.lower() == "csv":
            return pd.read_csv(file_path)
        elif format.lower() == "pickle":
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def list_files(self, pattern: str = "*") -> List[Path]:
        """List files matching pattern."""
        return list(self.base_dir.glob(pattern))
    
    def cleanup_old_files(self, pattern: str = "*", days_old: int = 7) -> int:
        """Remove files older than specified days."""
        import time
        
        current_time = time.time()
        cutoff_time = current_time - (days_old * 24 * 60 * 60)
        
        removed_count = 0
        for file_path in self.base_dir.glob(pattern):
            if file_path.is_file() and file_path.stat().st_mtime < cutoff_time:
                try:
                    file_path.unlink()
                    removed_count += 1
                except Exception as e:
                    console.print(f"❌ Error removing {file_path}: {e}")
        
        if removed_count > 0:
            console.print(f"🧹 Cleaned up {removed_count} old files")
        
        return removed_count


def merge_csv_files(input_files: List[Union[str, Path]], output_file: Union[str, Path]) -> bool:
    """Merge multiple CSV files into one."""
    try:
        dfs = []
        
        for file_path in input_files:
            df = read_csv_safe(file_path)
            if not df.empty:
                dfs.append(df)
        
        if not dfs:
            console.print("❌ No valid CSV files to merge")
            return False
        
        merged_df = pd.concat(dfs, ignore_index=True)
        return write_csv_safe(merged_df, output_file)
        
    except Exception as e:
        console.print(f"❌ Error merging CSV files: {e}")
        return False


def split_csv_file(input_file: Union[str, Path], output_dir: Union[str, Path], 
                  rows_per_file: int = 1000) -> List[Path]:
    """Split large CSV file into smaller files."""
    try:
        df = read_csv_safe(input_file)
        if df.empty:
            return []
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        input_path = Path(input_file)
        base_name = input_path.stem
        
        output_files = []
        
        for i, start_idx in enumerate(range(0, len(df), rows_per_file)):
            end_idx = start_idx + rows_per_file
            chunk = df.iloc[start_idx:end_idx]
            
            output_file = output_dir / f"{base_name}_part_{i+1:03d}.csv"
            if write_csv_safe(chunk, output_file):
                output_files.append(output_file)
        
        console.print(f"✅ Split {input_file} into {len(output_files)} files")
        return output_files
        
    except Exception as e:
        console.print(f"❌ Error splitting CSV file: {e}")
        return []


def validate_csv_structure(file_path: Union[str, Path], required_columns: List[str]) -> bool:
    """Validate CSV file has required columns."""
    try:
        df = read_csv_safe(file_path)
        if df.empty:
            return False
        
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            console.print(f"❌ Missing columns in {file_path}: {missing_columns}")
            return False
        
        return True
        
    except Exception as e:
        console.print(f"❌ Error validating CSV structure: {e}")
        return False


async def download_file_async(url: str, file_path: Union[str, Path]) -> bool:
    """Download file from URL asynchronously."""
    try:
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    async with aiofiles.open(file_path, 'wb') as f:
                        async for chunk in response.content.iter_chunked(8192):
                            await f.write(chunk)
                    
                    console.print(f"✅ Downloaded: {file_path}")
                    return True
                else:
                    console.print(f"❌ Download failed: HTTP {response.status}")
                    return False
                    
    except Exception as e:
        console.print(f"❌ Error downloading file: {e}")
        return False
