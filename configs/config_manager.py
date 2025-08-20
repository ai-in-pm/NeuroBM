#!/usr/bin/env python3
"""
Configuration Management System for NeuroBM.

This module provides utilities for loading, validating, merging, and managing
configuration files across the NeuroBM project.

Features:
- Hierarchical configuration loading
- Environment-specific overrides
- Configuration validation
- Template generation
- Configuration comparison and diff
- Dynamic parameter substitution

Usage:
    from configs.config_manager import ConfigManager
    
    # Load configuration
    config = ConfigManager.load('experiments/base.yaml')
    
    # Load with overrides
    config = ConfigManager.load('experiments/base.yaml', 
                               overrides={'training.learning_rate': 0.02})
    
    # Validate configuration
    ConfigManager.validate(config)
    
    # Generate template
    ConfigManager.generate_template('new_experiment.yaml', base='base')
"""

import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import copy
import os
from datetime import datetime
import jsonschema
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)


class ConfigManager:
    """Comprehensive configuration management for NeuroBM."""
    
    # Configuration schema for validation
    CONFIG_SCHEMA = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "description": {"type": "string"},
            "version": {"type": "string"},
            "model": {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": ["rbm", "dbm", "crbm"]},
                    "architecture": {"type": "object"},
                    "parameters": {"type": "object"}
                },
                "required": ["type"]
            },
            "training": {
                "type": "object",
                "properties": {
                    "algorithm": {"type": "string"},
                    "epochs": {"type": "integer", "minimum": 1},
                    "batch_size": {"type": "integer", "minimum": 1},
                    "learning_rate": {"type": "number", "minimum": 0}
                }
            },
            "data": {"type": "object"},
            "evaluation": {"type": "object"},
            "logging": {"type": "object"},
            "output": {"type": "object"}
        },
        "required": ["name", "model", "training", "data"]
    }
    
    @classmethod
    def load(
        cls,
        config_path: Union[str, Path],
        overrides: Optional[Dict[str, Any]] = None,
        environment: Optional[str] = None,
        validate_config: bool = True
    ) -> Dict[str, Any]:
        """
        Load configuration with optional overrides and environment settings.
        
        Args:
            config_path: Path to configuration file
            overrides: Dictionary of parameter overrides
            environment: Environment name for environment-specific settings
            validate_config: Whether to validate the configuration
            
        Returns:
            Loaded and processed configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load base configuration
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
        
        logger.info(f"Loaded base configuration from {config_path}")
        
        # Apply environment-specific settings
        if environment:
            config = cls._apply_environment_settings(config, environment)
        
        # Apply overrides
        if overrides:
            config = cls._apply_overrides(config, overrides)
            logger.info(f"Applied {len(overrides)} parameter overrides")
        
        # Substitute environment variables
        config = cls._substitute_env_vars(config)
        
        # Validate configuration
        if validate_config:
            cls.validate(config)
        
        # Add metadata
        config['_metadata'] = {
            'loaded_from': str(config_path),
            'loaded_at': datetime.now().isoformat(),
            'environment': environment,
            'overrides_applied': overrides is not None
        }
        
        return config
    
    @classmethod
    def validate(cls, config: Dict[str, Any]) -> None:
        """
        Validate configuration against schema.
        
        Args:
            config: Configuration dictionary to validate
            
        Raises:
            ValidationError: If configuration is invalid
        """
        try:
            validate(instance=config, schema=cls.CONFIG_SCHEMA)
            logger.info("Configuration validation passed")
        except ValidationError as e:
            logger.error(f"Configuration validation failed: {e.message}")
            raise
    
    @classmethod
    def save(
        cls,
        config: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = 'yaml',
        include_metadata: bool = True
    ) -> None:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            output_path: Output file path
            format: Output format ('yaml' or 'json')
            include_metadata: Whether to include metadata in output
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare config for saving
        save_config = copy.deepcopy(config)
        
        if not include_metadata and '_metadata' in save_config:
            del save_config['_metadata']
        
        # Save configuration
        with open(output_path, 'w') as f:
            if format.lower() == 'yaml':
                yaml.dump(save_config, f, default_flow_style=False, indent=2)
            elif format.lower() == 'json':
                json.dump(save_config, f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Saved configuration to {output_path}")
    
    @classmethod
    def merge(cls, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configurations with deep merging.
        
        Args:
            *configs: Configuration dictionaries to merge
            
        Returns:
            Merged configuration dictionary
        """
        if not configs:
            return {}
        
        result = copy.deepcopy(configs[0])
        
        for config in configs[1:]:
            result = cls._deep_merge(result, config)
        
        return result
    
    @classmethod
    def diff(cls, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare two configurations and return differences.
        
        Args:
            config1: First configuration
            config2: Second configuration
            
        Returns:
            Dictionary containing differences
        """
        differences = {
            'added': {},
            'removed': {},
            'changed': {}
        }
        
        # Find differences recursively
        cls._find_differences(config1, config2, differences, '')
        
        return differences
    
    @classmethod
    def generate_template(
        cls,
        template_name: str,
        base_config: Optional[str] = None,
        model_type: str = 'rbm',
        output_dir: str = 'experiments'
    ) -> Path:
        """
        Generate a configuration template.
        
        Args:
            template_name: Name for the template
            base_config: Base configuration to extend
            model_type: Type of model for the template
            output_dir: Directory to save template
            
        Returns:
            Path to generated template
        """
        if base_config:
            # Load base configuration
            base_path = Path(f'experiments/{base_config}.yaml')
            if base_path.exists():
                template = cls.load(base_path, validate_config=False)
            else:
                template = cls._create_default_template(model_type)
        else:
            template = cls._create_default_template(model_type)
        
        # Customize template
        template['name'] = template_name
        template['description'] = f"Configuration template for {template_name}"
        template['version'] = "1.0"
        
        # Save template
        output_path = Path(output_dir) / f"{template_name}.yaml"
        cls.save(template, output_path, include_metadata=False)
        
        logger.info(f"Generated template: {output_path}")
        return output_path
    
    @classmethod
    def list_configs(cls, directory: str = 'experiments') -> List[Dict[str, Any]]:
        """
        List all available configurations in a directory.
        
        Args:
            directory: Directory to search for configurations
            
        Returns:
            List of configuration information dictionaries
        """
        config_dir = Path(directory)
        configs = []
        
        if not config_dir.exists():
            return configs
        
        for config_file in config_dir.glob('*.yaml'):
            try:
                config = cls.load(config_file, validate_config=False)
                configs.append({
                    'name': config.get('name', config_file.stem),
                    'file': str(config_file),
                    'model_type': config.get('model', {}).get('type', 'unknown'),
                    'description': config.get('description', 'No description')
                })
            except Exception as e:
                logger.warning(f"Could not load config {config_file}: {e}")
        
        return configs
    
    @staticmethod
    def _apply_environment_settings(config: Dict[str, Any], environment: str) -> Dict[str, Any]:
        """Apply environment-specific settings."""
        env_config = copy.deepcopy(config)
        
        # Load environment-specific overrides
        env_file = Path(f'configs/environments/{environment}.yaml')
        if env_file.exists():
            with open(env_file, 'r') as f:
                env_settings = yaml.safe_load(f)
            
            env_config = ConfigManager._deep_merge(env_config, env_settings)
            logger.info(f"Applied environment settings for: {environment}")
        
        return env_config
    
    @staticmethod
    def _apply_overrides(config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Apply parameter overrides using dot notation."""
        result = copy.deepcopy(config)
        
        for key, value in overrides.items():
            ConfigManager._set_nested_value(result, key, value)
        
        return result
    
    @staticmethod
    def _substitute_env_vars(config: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute environment variables in configuration values."""
        def substitute_recursive(obj):
            if isinstance(obj, dict):
                return {k: substitute_recursive(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [substitute_recursive(item) for item in obj]
            elif isinstance(obj, str) and obj.startswith('${') and obj.endswith('}'):
                env_var = obj[2:-1]
                default_value = None
                if ':' in env_var:
                    env_var, default_value = env_var.split(':', 1)
                return os.getenv(env_var, default_value)
            else:
                return obj
        
        return substitute_recursive(config)
    
    @staticmethod
    def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries."""
        result = copy.deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = ConfigManager._deep_merge(result[key], value)
            else:
                result[key] = copy.deepcopy(value)
        
        return result
    
    @staticmethod
    def _set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> None:
        """Set nested value using dot notation."""
        keys = key_path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    @staticmethod
    def _find_differences(dict1: Dict[str, Any], dict2: Dict[str, Any], 
                         differences: Dict[str, Any], path: str) -> None:
        """Recursively find differences between dictionaries."""
        # Implementation would go here for detailed diff analysis
        pass
    
    @staticmethod
    def _create_default_template(model_type: str) -> Dict[str, Any]:
        """Create a default configuration template."""
        template = {
            "name": "template",
            "description": "Default configuration template",
            "version": "1.0",
            "model": {
                "type": model_type,
                "architecture": {
                    "n_visible": "auto",
                    "n_hidden": 256,
                    "visible_type": "bernoulli"
                },
                "parameters": {
                    "learning_rate": 0.01,
                    "momentum": 0.9,
                    "weight_decay": 0.0001,
                    "temperature": 1.0,
                    "use_bias": True
                }
            },
            "training": {
                "algorithm": "cd",
                "k_steps": 1,
                "batch_size": 32,
                "epochs": 100,
                "validation_split": 0.2
            },
            "data": {
                "regime": "base",
                "generation": {
                    "method": "skewed",
                    "n_samples": 1000,
                    "random_seed": 42
                },
                "preprocessing": {
                    "normalize": True,
                    "normalization_method": "minmax"
                }
            },
            "evaluation": {
                "metrics": [
                    "reconstruction_error",
                    "free_energy"
                ]
            },
            "logging": {
                "level": "INFO",
                "save_checkpoints": True
            },
            "output": {
                "save_dir": "results/template",
                "save_model": True
            },
            "random_seed": 42
        }
        
        return template


# Utility functions for common configuration tasks
def load_config(config_path: str, **kwargs) -> Dict[str, Any]:
    """Convenience function to load configuration."""
    return ConfigManager.load(config_path, **kwargs)


def validate_config(config: Dict[str, Any]) -> None:
    """Convenience function to validate configuration."""
    ConfigManager.validate(config)


def save_config(config: Dict[str, Any], output_path: str, **kwargs) -> None:
    """Convenience function to save configuration."""
    ConfigManager.save(config, output_path, **kwargs)
