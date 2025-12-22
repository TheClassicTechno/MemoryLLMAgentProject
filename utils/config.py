"""
Utility functions for configuration and file handling.
"""
import json
import yaml
from typing import Dict, Any
from pathlib import Path


def load_json_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to JSON config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_json(data: Dict[str, Any], output_path: str):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        output_path: Path to output file
    """
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def ensure_directory(dir_path: str):
    """
    Ensure directory exists, creating if necessary.
    
    Args:
        dir_path: Path to directory
    """
    Path(dir_path).mkdir(parents=True, exist_ok=True)


def get_config_path(filename: str) -> str:
    """
    Get path to a config file.
    
    Args:
        filename: Name of config file
        
    Returns:
        Full path to config file
    """
    config_dir = Path(__file__).parent.parent / 'configs'
    return str(config_dir / filename)


def get_results_path(filename: str = None) -> str:
    """
    Get path to results directory or file.
    
    Args:
        filename: Optional filename in results directory
        
    Returns:
        Path to results location
    """
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if filename:
        return str(results_dir / filename)
    return str(results_dir)
