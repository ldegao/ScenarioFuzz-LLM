"""
Utility functions for experiment runners.
"""

from pathlib import Path
import argparse


def validate_output_directory(output_root: str, parser: argparse.ArgumentParser) -> Path:
    """
    Validate and create output directory with write permission check.
    
    Args:
        output_root: Root directory path for experiment results
        parser: Argument parser for error reporting
        
    Returns:
        Path object for the validated output directory
        
    Raises:
        SystemExit: If directory cannot be created or is not writable
    """
    output_path = Path(output_root)
    
    # Check if parent directory exists and is writable
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Try to create a test file to verify write permissions
        test_file = output_path.parent / ".write_test"
        try:
            test_file.touch()
            test_file.unlink()
        except (OSError, PermissionError) as e:
            parser.error(f"Cannot write to output directory {output_root}: {e}")
    except (OSError, PermissionError) as e:
        parser.error(f"Invalid output directory path {output_root}: {e}")
    
    # Create output directory and test write permission
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        # Test write permission
        test_file = output_path / ".write_test"
        test_file.touch()
        test_file.unlink()
    except (OSError, PermissionError) as e:
        parser.error(f"Cannot write to output directory {output_root}: {e}")
    
    return output_path

