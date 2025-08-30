"""
File Utilities for Visionary

Provides file system operations, path handling utilities,
and batch file processing capabilities.
"""

import os
from pathlib import Path
from typing import List, Callable, Union

class FileUtils:
    @staticmethod
    def list_files(directory: str, extensions: List[str] = None) -> List[Path]:
        """
        List files in a directory with optional extension filtering.
        Args:
            directory: Path to directory
            extensions: List of file extensions to filter (e.g. ['.jpg', '.png'])
        Returns:
            List of Path objects for files
        """
        p = Path(directory)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        if extensions is None:
            files = [f for f in p.iterdir() if f.is_file()]
        else:
            files = [f for f in p.iterdir() if f.is_file() and f.suffix.lower() in extensions]
        return sorted(files)

    @staticmethod
    def create_dir(directory: str):
        """
        Create a directory if it does not exist.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def delete_file(path: str):
        """
        Delete a file if it exists.
        """
        p = Path(path)
        if p.exists() and p.is_file():
            p.unlink()

    @staticmethod
    def batch_process_files(files: List[Path], process_function: Callable, batch_size: int = 10) -> List:
        """
        Process files in batches.
        Args:
            files: List of file paths
            process_function: Function to process each batch
            batch_size: Number of files per batch
        Returns:
            List of results from processing
        """
        results = []
        for i in range(0, len(files), batch_size):
            batch = files[i:i+batch_size]
            batch_results = process_function(batch)
            results.extend(batch_results)
        return results

    @staticmethod
    def normalize_path(path: str) -> str:
        """
        Normalize a file path to absolute and expanded format.
        Args:
            path: Input file path
        Returns:
            Normalized absolute file path
        """
        p = Path(path).expanduser().resolve()
        return str(p)
def list_files_with_extensions(directory: str, extensions: Union[str, List[str]]) -> List[str]:
    """
    List files in a directory with specific extensions.
    
    Args:
        directory: Directory path to search
        extensions: File extension(s) to filter by (e.g., '.py' or ['.py', '.txt'])
    
    Returns:
        List of file paths matching the extensions
    """
    if isinstance(extensions, str):
        extensions = [extensions]
    
    # Ensure extensions start with a dot
    extensions = [ext if ext.startswith('.') else '.' + ext for ext in extensions]
    
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            if any(filename.lower().endswith(ext.lower()) for ext in extensions):
                files.append(os.path.join(root, filename))
    
    return files