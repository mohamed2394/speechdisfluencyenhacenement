import numpy as np
import librosa
from scipy import signal

class AudioPreprocessor:
    """Preprocessing module for audio enhancement and noise reduction."""
    
    def __init__(self, sample_rate=16000, frame_length=1024, hop_length=512,
                 noise_reduce_strength=0.2, normalize_audio=True):
        """
        Initialize the audio preprocessor.
        
        Parameters:
        -----------
        sample_rate : int
            Audio sample rate
        frame_length : int
            Length of each frame for processing
        hop_length : int
            Number of samples between frames
        noise_reduce_strength : float
            Strength of noise reduction (0.0 to 1.0)
        normalize_audio : bool
            Whether to normalize audio after processing
        """
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.hop_length = hop_length
        self.noise_reduce_strength = noise_reduce_strength
        self.normalize_audio = normalize_audio
    
    def process(self, audio: np.ndarray) -> np.ndarray:
        """Apply all preprocessing steps to the audio."""
        # Apply noise reduction
        audio = self._reduce_noise(audio)
        
        # Apply bandpass filter to focus on speech frequencies (80Hz - 8000Hz)
        audio = self._apply_bandpass(audio)
        
        # Normalize if requested
        if self.normalize_audio:
            audio = self._normalize(audio)
        
        return audio
    
    def _reduce_noise(self, audio: np.ndarray) -> np.ndarray:
        """Reduce background noise using spectral gating."""
        # Compute spectrogram
        D = librosa.stft(audio, n_fft=self.frame_length, hop_length=self.hop_length)
        mag, phase = librosa.magphase(D)
        
        # Estimate noise profile from lowest energy frames
        noise_profile = np.mean(np.sort(mag, axis=1)[:, :int(mag.shape[1] * 0.1)], axis=1)
        noise_profile = noise_profile.reshape(-1, 1)
        
        # Apply soft thresholding
        threshold = noise_profile * (1 + self.noise_reduce_strength)
        mask = (mag - threshold) / mag
        mask = np.maximum(0, mask)
        mask = np.minimum(1, mask)
        
        # Apply mask and reconstruct
        mag_denoised = mag * mask
        D_denoised = mag_denoised * phase
        audio_denoised = librosa.istft(D_denoised, hop_length=self.hop_length)
        
        return audio_denoised
    
    def _apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply bandpass filter focused on speech frequencies."""
        nyquist = self.sample_rate // 2
        low = 80 / nyquist
        high = 8000 / nyquist
        b, a = signal.butter(4, [low, high], btype='band')
        return signal.filtfilt(b, a, audio)
    
    def _normalize(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to a consistent volume level."""
        # Peak normalization
        peak = np.abs(audio).max()
        if peak > 0:
            audio = audio / peak * 0.9  # Leave some headroom
        
        return audio
    
    def analyze_signal_quality(self, audio: np.ndarray) -> dict:
        """Analyze various aspects of signal quality."""
        metrics = {}
        
        # Signal-to-noise ratio estimation
        noise_floor = np.sort(np.abs(audio))[:int(len(audio) * 0.1)].mean()
        signal_level = np.abs(audio).mean()
        metrics['estimated_snr'] = 20 * np.log10(signal_level / noise_floor) if noise_floor > 0 else float('inf')
        
        # Peak level
        metrics['peak_level'] = np.abs(audio).max()
        
        # RMS level
        metrics['rms_level'] = np.sqrt(np.mean(audio ** 2))
        
        # Spectral centroid (brightness)
        metrics['spectral_centroid'] = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate
        ).mean()
        
        return metrics