import json
import os
from typing import Dict, Any, Optional

class ConfigManager:
    """Configuration management system for the disfluency enhancement pipeline."""
    
    DEFAULT_CONFIG = {
        'preprocessing': {
            'sample_rate': 16000,
            'frame_length': 1024,
            'hop_length': 512,
            'noise_reduce_strength': 0.2,
            'normalize_audio': True
        },
        'disfluency_enhancer': {
            'silence_thresh': 0.01,
            'min_silence_duration': 0.1,
            'prolongation_thresh': 0.85,
            'prolongation_window': 0.3
        },
        'cluttering_enhancer': {
            'n_fft': 512,
            'hop_length': 128,
            'model_path': 'models/cluttering_model.pth'
        },
        'triage': {
            'classifier_path': 'models/disfluency_classifier.joblib',
            'tempo_threshold': 160,
            'rhythm_regularity_threshold': 0.5,
            'energy_variance_threshold': 0.01
        },
        'performance_monitoring': {
            'log_level': 'INFO',
            'metrics_save_path': 'metrics/',
            'save_intermediate_results': False
        }
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to configuration file
        """
        self.config_path = config_path
        self.config = self.DEFAULT_CONFIG.copy()
        
        if config_path and os.path.exists(config_path):
            self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                loaded_config = json.load(f)
                self._update_nested_dict(self.config, loaded_config)
        except Exception as e:
            print(f"Error loading config: {e}")
            print("Using default configuration")
    
    def save_config(self, config_path: Optional[str] = None) -> None:
        """Save current configuration to file.
        
        Parameters:
        -----------
        config_path : str, optional
            Path to save configuration file
        """
        save_path = config_path or self.config_path
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                json.dump(self.config, f, indent=4)
    
    def get_config(self, section: Optional[str] = None) -> Dict[str, Any]:
        """Get configuration or section of configuration.
        
        Parameters:
        -----------
        section : str, optional
            Section name to retrieve
            
        Returns:
        --------
        dict
            Configuration dictionary
        """
        if section:
            return self.config.get(section, {})
        return self.config
    
    def update_config(self, updates: Dict[str, Any], section: Optional[str] = None) -> None:
        """Update configuration values.
        
        Parameters:
        -----------
        updates : dict
            Dictionary of updates to apply
        section : str, optional
            Section to update
        """
        if section:
            if section not in self.config:
                self.config[section] = {}
            self._update_nested_dict(self.config[section], updates)
        else:
            self._update_nested_dict(self.config, updates)
    
    def _update_nested_dict(self, d: Dict[str, Any], u: Dict[str, Any]) -> None:
        """Recursively update nested dictionary."""
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                self._update_nested_dict(d[k], v)
            else:
                d[k] = v
    
    def validate_config(self) -> bool:
        """Validate current configuration.
        
        Returns:
        --------
        bool
            True if configuration is valid
        """
        try:
            # Validate preprocessing settings
            assert self.config['preprocessing']['sample_rate'] > 0
            assert self.config['preprocessing']['frame_length'] > 0
            assert self.config['preprocessing']['hop_length'] > 0
            assert 0 <= self.config['preprocessing']['noise_reduce_strength'] <= 1
            
            # Validate disfluency enhancer settings
            assert self.config['disfluency_enhancer']['silence_thresh'] > 0
            assert self.config['disfluency_enhancer']['min_silence_duration'] > 0
            assert 0 < self.config['disfluency_enhancer']['prolongation_thresh'] <= 1
            assert self.config['disfluency_enhancer']['prolongation_window'] > 0
            
            # Validate cluttering enhancer settings
            assert self.config['cluttering_enhancer']['n_fft'] > 0
            assert self.config['cluttering_enhancer']['hop_length'] > 0
            
            # Validate triage settings
            assert self.config['triage']['tempo_threshold'] > 0
            assert self.config['triage']['rhythm_regularity_threshold'] > 0
            assert self.config['triage']['energy_variance_threshold'] > 0
            
            return True
        except (KeyError, AssertionError):
            return False