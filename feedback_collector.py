import numpy as np
import librosa
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional

class FeedbackCollector:
    """Collects and manages feedback data for model improvement."""
    
    def __init__(self, metrics_log_path: str = 'enhancement_metrics.log'):
        """Initialize the feedback collector.
        
        Parameters:
        -----------
        metrics_log_path : str
            Path to save metrics log
        """
        self.metrics_log_path = metrics_log_path
        self._setup_logging()
        
    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        logging.basicConfig(
            filename=self.metrics_log_path,
            level=logging.INFO,
            format='%(asctime)s - %(message)s'
        )
    
    def collect_metrics(self, original_audio: np.ndarray, enhanced_audio: np.ndarray,
                       sample_rate: int, enhancement_type: str) -> Dict[str, Any]:
        """Collect audio quality metrics.
        
        Parameters:
        -----------
        original_audio : numpy.ndarray
            Original audio signal
        enhanced_audio : numpy.ndarray
            Enhanced audio signal
        sample_rate : int
            Audio sample rate
        enhancement_type : str
            Type of enhancement applied ('stuttering' or 'cluttering')
            
        Returns:
        --------
        dict
            Collected metrics
        """
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'enhancement_type': enhancement_type,
            'audio_length': len(original_audio) / sample_rate,
            'metrics': {}
        }
        
        # Calculate signal-to-noise ratio
        metrics['metrics']['snr_before'] = self._calculate_snr(original_audio)
        metrics['metrics']['snr_after'] = self._calculate_snr(enhanced_audio)
        
        # Calculate speech rate (syllables per second)
        metrics['metrics']['speech_rate_before'] = self._estimate_speech_rate(original_audio, sample_rate)
        metrics['metrics']['speech_rate_after'] = self._estimate_speech_rate(enhanced_audio, sample_rate)
        
        # Calculate spectral contrast (measure of clarity)
        metrics['metrics']['spectral_contrast_before'] = self._calculate_spectral_contrast(original_audio, sample_rate)
        metrics['metrics']['spectral_contrast_after'] = self._calculate_spectral_contrast(enhanced_audio, sample_rate)
        
        # Log metrics
        self._log_metrics(metrics)
        
        return metrics
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate signal-to-noise ratio."""
        noise_floor = np.sort(np.abs(audio))[:int(len(audio) * 0.1)].mean()
        signal = np.abs(audio).mean()
        return 20 * np.log10(signal / noise_floor) if noise_floor > 0 else 100.0
    
    def _estimate_speech_rate(self, audio: np.ndarray, sample_rate: int) -> float:
        """Estimate speech rate using onset detection."""
        onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
        onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env)
        
        # Estimate syllables from onsets (rough approximation)
        duration = len(audio) / sample_rate
        return len(onset_frames) / duration if duration > 0 else 0.0
    
    def _calculate_spectral_contrast(self, audio: np.ndarray, sample_rate: int) -> float:
        """Calculate average spectral contrast."""
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        return float(np.mean(contrast))
    
    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to file."""
        logging.info(json.dumps(metrics))
    
    def analyze_feedback(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Analyze collected feedback data.
        
        Parameters:
        -----------
        time_window : float, optional
            Time window in hours to analyze (None for all data)
            
        Returns:
        --------
        dict
            Analysis results
        """
        metrics_data = self._load_metrics(time_window)
        
        if not metrics_data:
            return {'error': 'No data available for analysis'}
        
        analysis = {
            'total_samples': len(metrics_data),
            'enhancement_types': {},
            'average_improvements': {
                'snr': 0.0,
                'speech_rate': 0.0,
                'spectral_contrast': 0.0
            }
        }
        
        # Analyze by enhancement type
        for entry in metrics_data:
            enhancement_type = entry['enhancement_type']
            if enhancement_type not in analysis['enhancement_types']:
                analysis['enhancement_types'][enhancement_type] = 0
            analysis['enhancement_types'][enhancement_type] += 1
            
            # Calculate improvements
            metrics = entry['metrics']
            analysis['average_improvements']['snr'] += (
                metrics['snr_after'] - metrics['snr_before']
            )
            analysis['average_improvements']['speech_rate'] += (
                metrics['speech_rate_after'] - metrics['speech_rate_before']
            )
            analysis['average_improvements']['spectral_contrast'] += (
                metrics['spectral_contrast_after'] - metrics['spectral_contrast_before']
            )
        
        # Calculate averages
        for metric in analysis['average_improvements']:
            analysis['average_improvements'][metric] /= len(metrics_data)
        
        return analysis
    
    def _load_metrics(self, time_window: Optional[float] = None) -> list:
        """Load metrics from log file."""
        metrics_data = []
        current_time = datetime.now()
        
        try:
            with open(self.metrics_log_path, 'r') as f:
                for line in f:
                    try:
                        # Extract JSON from log line
                        json_str = line.split(' - ', 1)[1]
                        entry = json.loads(json_str)
                        
                        # Apply time window filter if specified
                        if time_window is not None:
                            entry_time = datetime.fromisoformat(entry['timestamp'])
                            hours_diff = (current_time - entry_time).total_seconds() / 3600
                            if hours_diff > time_window:
                                continue
                        
                        metrics_data.append(entry)
                    except Exception as e:
                        print(f"Error parsing log entry: {e}")
        except FileNotFoundError:
            print(f"Metrics log file not found: {self.metrics_log_path}")
        
        return metrics_data