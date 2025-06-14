import os
import numpy as np
import torch
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from audio_preprocessor import AudioPreprocessor
from feedback_collector import FeedbackCollector
from real_time_processor import RealTimeProcessor

class BatchProcessor:
    """Handles batch processing of audio files for disfluency enhancement."""
    
    def __init__(self, config: Dict[str, Any], num_workers: int = 4):
        """Initialize batch processor.
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        num_workers : int
            Number of worker threads for parallel processing
        """
        self.config = config
        self.num_workers = num_workers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize components
        self.preprocessor = AudioPreprocessor(**config['preprocessing'])
        self.feedback_collector = FeedbackCollector(
            metrics_log_path=config['feedback']['metrics_log_path']
        )
        
    def process_batch(self, input_files: List[str], output_dir: str,
                      enhancement_type: Optional[str] = None) -> Dict[str, Any]:
        """Process a batch of audio files.
        
        Parameters:
        -----------
        input_files : list
            List of input audio file paths
        output_dir : str
            Directory to save enhanced audio files
        enhancement_type : str, optional
            Force specific enhancement type ('stuttering' or 'cluttering')
            
        Returns:
        --------
        dict
            Processing results and metrics
        """
        os.makedirs(output_dir, exist_ok=True)
        results = {'processed_files': [], 'metrics': {}}
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = []
            for input_file in input_files:
                future = executor.submit(
                    self._process_single_file,
                    input_file,
                    output_dir,
                    enhancement_type
                )
                futures.append(future)
            
            # Collect results
            for future in futures:
                try:
                    result = future.result()
                    results['processed_files'].append(result)
                except Exception as e:
                    print(f"Error processing file: {e}")
        
        # Aggregate metrics
        if self.config['feedback']['collect_metrics']:
            results['metrics'] = self._aggregate_metrics(results['processed_files'])
        
        return results
    
    def _process_single_file(self, input_file: str, output_dir: str,
                            enhancement_type: Optional[str]) -> Dict[str, Any]:
        """Process a single audio file.
        
        Parameters:
        -----------
        input_file : str
            Input audio file path
        output_dir : str
            Output directory
        enhancement_type : str, optional
            Forced enhancement type
            
        Returns:
        --------
        dict
            Processing results for the file
        """
        result = {
            'input_file': input_file,
            'output_file': None,
            'enhancement_type': enhancement_type,
            'metrics': None,
            'status': 'failed',
            'error': None
        }
        
        try:
            # Generate output filename
            filename = os.path.basename(input_file)
            base, ext = os.path.splitext(filename)
            output_file = os.path.join(output_dir, f"{base}_enhanced{ext}")
            result['output_file'] = output_file
            
            # Process file using hybrid enhancer
            from disfluency_hybrid_system import HybridDisfluencyEnhancer
            enhancer = HybridDisfluencyEnhancer(
                classifier_path=self.config['model_paths']['classifier'],
                cluttering_model_path=self.config['model_paths']['cluttering_model']
            )
            
            # Apply enhancement
            enhancer.enhance(input_file, output_file, force_type=enhancement_type)
            
            # Collect metrics if enabled
            if self.config['feedback']['collect_metrics']:
                import librosa
                original_audio, sr = librosa.load(input_file, sr=None)
                enhanced_audio, _ = librosa.load(output_file, sr=sr)
                
                metrics = self.feedback_collector.collect_metrics(
                    original_audio,
                    enhanced_audio,
                    sr,
                    enhancement_type or 'auto'
                )
                result['metrics'] = metrics
            
            result['status'] = 'success'
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def _aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate metrics from multiple processed files.
        
        Parameters:
        -----------
        results : list
            List of processing results
            
        Returns:
        --------
        dict
            Aggregated metrics
        """
        aggregated = {
            'total_files': len(results),
            'successful_files': sum(1 for r in results if r['status'] == 'success'),
            'failed_files': sum(1 for r in results if r['status'] == 'failed'),
            'average_improvements': {
                'snr': 0.0,
                'speech_rate': 0.0,
                'spectral_contrast': 0.0
            }
        }
        
        # Calculate average improvements
        success_count = aggregated['successful_files']
        if success_count > 0:
            for result in results:
                if result['status'] == 'success' and result['metrics']:
                    metrics = result['metrics']['metrics']
                    for key in aggregated['average_improvements']:
                        before_key = f"{key}_before"
                        after_key = f"{key}_after"
                        if before_key in metrics and after_key in metrics:
                            improvement = metrics[after_key] - metrics[before_key]
                            aggregated['average_improvements'][key] += improvement
            
            # Calculate averages
            for key in aggregated['average_improvements']:
                aggregated['average_improvements'][key] /= success_count
        
        return aggregated