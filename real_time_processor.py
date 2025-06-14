import numpy as np
import sounddevice as sd
import queue
import threading
from typing import Optional, Callable
from audio_preprocessor import AudioPreprocessor

class RealTimeProcessor:
    """Handles real-time audio processing for disfluency enhancement."""
    
    def __init__(self, sample_rate: int = 16000, buffer_size: int = 1024,
                 channels: int = 1, dtype=np.float32):
        """Initialize real-time processor.
        
        Parameters:
        -----------
        sample_rate : int
            Audio sample rate
        buffer_size : int
            Size of audio buffer for processing
        channels : int
            Number of audio channels
        dtype : type
            Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.channels = channels
        self.dtype = dtype
        
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.preprocessor = AudioPreprocessor(sample_rate=sample_rate)
        
        self._running = False
        self._processing_thread = None
        self._stream = None
        
    def start(self, processing_callback: Optional[Callable] = None) -> None:
        """Start real-time processing.
        
        Parameters:
        -----------
        processing_callback : callable, optional
            Custom processing function to apply to audio chunks
        """
        if self._running:
            print("Real-time processing already running")
            return
        
        self._running = True
        
        # Start processing thread
        self._processing_thread = threading.Thread(
            target=self._process_audio,
            args=(processing_callback,)
        )
        self._processing_thread.daemon = True
        self._processing_thread.start()
        
        # Start audio stream
        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            blocksize=self.buffer_size,
            channels=self.channels,
            dtype=self.dtype,
            callback=self._audio_callback
        )
        self._stream.start()
        
        print("Real-time processing started")
    
    def stop(self) -> None:
        """Stop real-time processing."""
        if not self._running:
            return
        
        self._running = False
        
        # Stop audio stream
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Wait for processing thread to finish
        if self._processing_thread is not None:
            self._processing_thread.join()
            self._processing_thread = None
        
        print("Real-time processing stopped")
    
    def _audio_callback(self, indata: np.ndarray, outdata: np.ndarray,
                       frames: int, time, status) -> None:
        """Callback for audio stream processing."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Put input data in queue
        self.input_queue.put(indata.copy())
        
        try:
            # Get processed data from output queue
            outdata[:] = self.output_queue.get_nowait()
        except queue.Empty:
            # If no processed data available, output silence
            outdata.fill(0)
    
    def _process_audio(self, processing_callback: Optional[Callable]) -> None:
        """Process audio chunks from input queue."""
        overlap_buffer = np.zeros((self.buffer_size, self.channels), dtype=self.dtype)
        overlap_size = self.buffer_size // 2
        
        while self._running:
            try:
                # Get input chunk
                chunk = self.input_queue.get(timeout=0.1)
                
                # Combine with previous overlap
                processing_frame = np.concatenate([overlap_buffer, chunk])
                
                # Preprocess audio
                processed_frame = self.preprocessor.process(processing_frame)
                
                # Apply custom processing if provided
                if processing_callback is not None:
                    processed_frame = processing_callback(processed_frame)
                
                # Update overlap buffer and put processed chunk in output queue
                overlap_buffer = processed_frame[-overlap_size:]
                self.output_queue.put(processed_frame[:-overlap_size])
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
    
    def get_input_latency(self) -> float:
        """Get input stream latency."""
        return self._stream.input_latency if self._stream else 0.0
    
    def get_output_latency(self) -> float:
        """Get output stream latency."""
        return self._stream.output_latency if self._stream else 0.0
    
    def is_running(self) -> bool:
        """Check if processing is running."""
        return self._running