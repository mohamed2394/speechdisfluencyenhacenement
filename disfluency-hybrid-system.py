import numpy as np
import librosa
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import joblib
import os

class DisfluencyEnhancer:
    """
    Enhancer for speech disfluencies focusing on pause removal and prolongation smoothing.
    """
    def __init__(self, sample_rate=16000, silence_thresh=0.01, min_silence_duration=0.1,
                 prolongation_thresh=0.85, prolongation_window=0.3):
        """
        Parameters:
        -----------
        sample_rate : int
            Audio sample rate
        silence_thresh : float
            Amplitude threshold below which frames are considered silence
        min_silence_duration : float
            Minimum duration (in seconds) of silence to be removed
        prolongation_thresh : float
            Similarity threshold for detecting prolonged sounds
        prolongation_window : float
            Window size (in seconds) for analyzing prolongations
        """
        self.sample_rate = sample_rate
        self.silence_thresh = silence_thresh
        self.min_silence_frames = int(min_silence_duration * sample_rate)
        self.prolongation_thresh = prolongation_thresh
        self.prolongation_window_frames = int(prolongation_window * sample_rate)
        
    def enhance(self, audio_path: str, output_path: str):
        """
        Load audio, apply pause removal and prolongation smoothing, and save result.
        """
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        # Step 1: Remove long pauses
        audio_nopause = self._remove_pauses(audio)
        # Step 2: Smooth prolongations
        audio_smoothed = self._smooth_prolongations(audio_nopause)
        # Save enhanced audio
        sf.write(output_path, audio_smoothed, sr)
        print(f"Enhanced audio saved to {output_path}")
        
    def _remove_pauses(self, audio: np.ndarray) -> np.ndarray:
        """
        Remove segments of silence longer than the minimum duration.
        """
        # Calculate RMS energy for the entire signal
        frame_length = 1024
        hop_length = 512
        
        # Calculate energy for each frame
        energy = np.array([
            np.sqrt(np.mean(audio[i:i+frame_length]**2)) 
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        # Mark frames as silent or not
        is_silent = energy < self.silence_thresh
        
        # Find continuous silence regions
        silence_regions = []
        current_silence_start = None
        
        for i, silent in enumerate(is_silent):
            frame_pos = i * hop_length
            
            if silent and current_silence_start is None:
                current_silence_start = frame_pos
            elif not silent and current_silence_start is not None:
                silence_duration = frame_pos - current_silence_start
                if silence_duration >= self.min_silence_frames:
                    silence_regions.append((current_silence_start, frame_pos))
                current_silence_start = None
                
        # Also check if we ended in silence
        if current_silence_start is not None:
            silence_duration = len(audio) - current_silence_start
            if silence_duration >= self.min_silence_frames:
                silence_regions.append((current_silence_start, len(audio)))
        
        # Create a new audio array without the long silences
        if not silence_regions:
            return audio
            
        result = []
        last_end = 0
        
        for start, end in silence_regions:
            # Add audio up to the silence
            result.append(audio[last_end:start])
            # Add a shorter version of the silence (keep some silence for natural speech)
            short_silence_length = int(0.05 * self.sample_rate)  # 50ms of silence
            if short_silence_length > 0:
                result.append(np.zeros(short_silence_length))
            last_end = end
            
        # Add remaining audio after last silence
        if last_end < len(audio):
            result.append(audio[last_end:])
            
        return np.concatenate(result)
    
    def _smooth_prolongations(self, audio: np.ndarray) -> np.ndarray:
        """
        Attenuate or taper prolonged sounds to smooth out sustained phonemes.
        """
        # Extract MFCC features to detect spectral stability (indicating prolongations)
        hop_length = 512
        n_mfcc = 13
        
        # Calculate MFCCs
        mfccs = librosa.feature.mfcc(
            y=audio, 
            sr=self.sample_rate, 
            n_mfcc=n_mfcc,
            hop_length=hop_length
        )
        
        # Calculate frame-to-frame similarity to detect prolongations
        # High similarity over consecutive frames indicates prolonged sounds
        frame_similarities = []
        for i in range(1, mfccs.shape[1]):
            # Cosine similarity between consecutive frames
            similarity = np.dot(mfccs[:, i], mfccs[:, i-1]) / (
                np.linalg.norm(mfccs[:, i]) * np.linalg.norm(mfccs[:, i-1])
            )
            frame_similarities.append(similarity)
            
        frame_similarities = np.array(frame_similarities)
        
        # Detect prolongations (segments with high consecutive similarity)
        prolongation_regions = []
        current_prolongation_start = None
        
        for i, similarity in enumerate(frame_similarities):
            frame_pos = (i + 1) * hop_length  # +1 because we start at second frame
            
            if similarity > self.prolongation_thresh and current_prolongation_start is None:
                current_prolongation_start = frame_pos
            elif (similarity < self.prolongation_thresh or i == len(frame_similarities) - 1) and current_prolongation_start is not None:
                # Check if prolongation is long enough (at least 200ms)
                prolongation_duration = frame_pos - current_prolongation_start
                if prolongation_duration >= 0.2 * self.sample_rate:
                    prolongation_regions.append((current_prolongation_start, frame_pos))
                current_prolongation_start = None
        
        # Apply smoothing to prolonged regions
        result = np.copy(audio)
        
        for start, end in prolongation_regions:
            # Ensure indices are within bounds
            start = max(0, start)
            end = min(len(audio), end)
            
            if end <= start:
                continue
                
            # Extract the prolonged segment
            segment = audio[start:end]
            
            # For stuttering prolongations, apply a tapering window to smooth
            # Create a custom decay function
            decay_env = np.linspace(1.0, 0.3, len(segment))
            decay_env = np.power(decay_env, 0.5)  # Make decay less aggressive
            
            # Apply decay
            smoothed_segment = segment * decay_env
            
            # Blend back into the original with a crossfade for natural sound
            crossfade_len = min(int(0.05 * self.sample_rate), len(segment) // 4)
            if crossfade_len > 0:
                # Start crossfade
                fade_in = np.linspace(0, 1, crossfade_len)
                smoothed_segment[:crossfade_len] = (
                    segment[:crossfade_len] * (1 - fade_in) + 
                    smoothed_segment[:crossfade_len] * fade_in
                )
                
                # End crossfade
                fade_out = np.linspace(1, 0, crossfade_len)
                smoothed_segment[-crossfade_len:] = (
                    segment[-crossfade_len:] * (1 - fade_out) + 
                    smoothed_segment[-crossfade_len:] * fade_out
                )
            
            # Replace the segment in the result
            result[start:end] = smoothed_segment
            
        return result


class ClutteringEnhancerModel(nn.Module):
    """
    Neural network model for enhancing cluttered speech.
    Based on a modified U-Net architecture for audio enhancement.
    """
    def __init__(self, n_fft=512, hop_length=128):
        super(ClutteringEnhancerModel, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Encoder layers
        self.enc1 = nn.Conv1d(2, 16, kernel_size=7, padding=3)  # Input: mag + phase
        self.enc2 = nn.Conv1d(16, 32, kernel_size=7, padding=3, stride=2)
        self.enc3 = nn.Conv1d(32, 64, kernel_size=7, padding=3, stride=2)
        self.enc4 = nn.Conv1d(64, 128, kernel_size=7, padding=3, stride=2)
        
        # Context module with dilation to capture wider contexts
        self.context1 = nn.Conv1d(128, 128, kernel_size=7, padding=6, dilation=2)
        self.context2 = nn.Conv1d(128, 128, kernel_size=7, padding=12, dilation=4)
        self.context3 = nn.Conv1d(128, 128, kernel_size=7, padding=18, dilation=6)
        
        # Decoder layers
        self.dec4 = nn.ConvTranspose1d(256, 64, kernel_size=8, stride=2, padding=3)
        self.dec3 = nn.ConvTranspose1d(128, 32, kernel_size=8, stride=2, padding=3)
        self.dec2 = nn.ConvTranspose1d(64, 16, kernel_size=8, stride=2, padding=3)
        self.dec1 = nn.Conv1d(32, 2, kernel_size=7, padding=3)  # Output: mag + phase
        
        # Normalization layers
        self.norm1 = nn.BatchNorm1d(16)
        self.norm2 = nn.BatchNorm1d(32)
        self.norm3 = nn.BatchNorm1d(64)
        self.norm4 = nn.BatchNorm1d(128)
        
    def forward(self, x):
        # Encoder
        e1 = F.relu(self.norm1(self.enc1(x)))
        e2 = F.relu(self.norm2(self.enc2(e1)))
        e3 = F.relu(self.norm3(self.enc3(e2)))
        e4 = F.relu(self.norm4(self.enc4(e3)))
        
        # Context processing
        c1 = F.relu(self.context1(e4))
        c2 = F.relu(self.context2(c1))
        c3 = F.relu(self.context3(c2))
        
        # Skip connections + decoder
        d4 = F.relu(self.dec4(torch.cat([c3, e4], dim=1)))
        d3 = F.relu(self.dec3(torch.cat([d4, e3], dim=1)))
        d2 = F.relu(self.dec2(torch.cat([d3, e2], dim=1)))
        d1 = self.dec1(torch.cat([d2, e1], dim=1))
        
        return d1
    
    def stft(self, audio):
        """Convert audio to STFT domain"""
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude, phase = librosa.magphase(stft)
        # Log magnitude
        magnitude = np.log1p(magnitude)
        # Phase to angle
        phase = np.angle(phase)
        
        # Stack magnitude and phase
        features = np.stack([magnitude, phase], axis=0)
        return features, stft.shape
    
    def istft(self, features, stft_shape):
        """Convert STFT domain back to audio"""
        magnitude = features[0]
        phase = features[1]
        
        # Convert log magnitude back
        magnitude = np.exp(magnitude) - 1.0
        
        # Reconstruct complex STFT
        stft_matrix = magnitude * np.exp(1j * phase)
        
        # Convert back to time domain
        audio = librosa.istft(stft_matrix, hop_length=self.hop_length)
        return audio


class ClutteringEnhancer:
    """
    ML-based enhancer for cluttered speech, focusing on improving rhythm,
    articulation, and overall speech intelligibility.
    """
    def __init__(self, model_path=None, sample_rate=16000):
        """
        Initialize the cluttering enhancer with a pre-trained model.
        
        Parameters:
        -----------
        model_path : str, optional
            Path to pre-trained model file
        sample_rate : int
            Audio sample rate
        """
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the enhancer model
        self.model = ClutteringEnhancerModel().to(self.device)
        
        # Load pre-trained model if provided
        if model_path and os.path.exists(model_path):
            print(f"Loading cluttering enhancement model from {model_path}")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
        else:
            print("No cluttering model provided. Using untrained model.")
    
    def enhance(self, audio_path: str, output_path: str):
        """
        Enhance cluttered speech and save result.
        
        Parameters:
        -----------
        audio_path : str
            Path to input audio file
        output_path : str
            Path to save enhanced audio
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        # Process using neural network model
        enhanced_audio = self._enhance_speech(audio)
        
        # Save enhanced audio
        sf.write(output_path, enhanced_audio, self.sample_rate)
        print(f"Enhanced audio saved to {output_path}")
    
    def _enhance_speech(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply neural network enhancement to audio.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
            
        Returns:
        --------
        numpy.ndarray
            Enhanced audio signal
        """
        # Convert to torch tensor and move to device
        with torch.no_grad():
            # Convert to STFT domain
            features, stft_shape = self.model.stft(audio)
            
            # Convert to torch tensor
            features_tensor = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            
            # Apply model
            enhanced_features = self.model(features_tensor).squeeze(0).cpu().numpy()
            
            # Convert back to time domain
            enhanced_audio = self.model.istft(enhanced_features, stft_shape)
            
            return enhanced_audio
    
    def _adjust_speech_rate(self, audio: np.ndarray) -> np.ndarray:
        """
        Adjusts speech rate for enhanced intelligibility.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
            
        Returns:
        --------
        numpy.ndarray
            Rate-adjusted audio
        """
        # Detect speech tempo using librosa beat tracking
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # If speech is too fast (over 180 bpm), slow it down
        if tempo > 180:
            # Calculate stretch factor to bring tempo closer to 150 bpm (clear speech rate)
            stretch_factor = tempo / 150
            
            # Time-stretch audio using phase vocoder
            stretched_audio = librosa.effects.time_stretch(audio, rate=stretch_factor)
            return stretched_audio
        
        return audio


class DisfluencyTriage:
    """
    Triage system to detect and classify speech disfluencies into stuttering or cluttering
    to route to the appropriate enhancer.
    """
    def __init__(self, classifier_path=None):
        """
        Initialize the triage system.
        
        Parameters:
        -----------
        classifier_path : str, optional
            Path to pre-trained classifier model
        """
        self.sample_rate = 16000
        self.classifier = None
        
        # Load classifier if provided
        if classifier_path and os.path.exists(classifier_path):
            print(f"Loading disfluency classifier from {classifier_path}")
            self.classifier = joblib.load(classifier_path)
    
    def classify(self, audio_path: str):
        """
        Classify audio as stuttering or cluttering.
        
        Parameters:
        -----------
        audio_path : str
            Path to audio file
            
        Returns:
        --------
        str
            Classification result ("stuttering" or "cluttering")
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate)
        
        if self.classifier:
            # Extract features for classification
            features = self._extract_features(audio)
            
            # Use trained classifier
            prediction = self.classifier.predict([features])[0]
            return prediction
        else:
            # Use heuristic classification if no trained model
            return self._heuristic_classification(audio)
    
    def _extract_features(self, audio: np.ndarray) -> list:
        """
        Extract features for classification.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
            
        Returns:
        --------
        list
            Feature vector
        """
        # Extract MFCCs
        mfccs = librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
        mfcc_means = np.mean(mfccs, axis=1)
        mfcc_vars = np.var(mfccs, axis=1)
        
        # Extract spectral contrast
        contrast = librosa.feature.spectral_contrast(y=audio, sr=self.sample_rate)
        contrast_means = np.mean(contrast, axis=1)
        
        # Extract rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Extract energy variance (high for stuttering, lower for cluttering)
        rms = librosa.feature.rms(y=audio)[0]
        rms_var = np.var(rms)
        
        # Combine all features
        features = list(mfcc_means) + list(mfcc_vars) + list(contrast_means) + [tempo, rms_var]
        
        return features
    
    def _heuristic_classification(self, audio: np.ndarray) -> str:
        """
        Simple heuristic classification based on audio characteristics.
        
        Parameters:
        -----------
        audio : numpy.ndarray
            Audio signal
            
        Returns:
        --------
        str
            Classification result ("stuttering" or "cluttering")
        """
        # Extract rhythm features
        onset_env = librosa.onset.onset_strength(y=audio, sr=self.sample_rate)
        tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sample_rate)
        
        # Extract energy features
        rms = librosa.feature.rms(y=audio)[0]
        
        # Calculate energy variance (high for stuttering due to blocks and repetitions)
        rms_var = np.var(rms)
        
        # Calculate rhythm regularity (lower for cluttering)
        if len(beats) > 1:
            beat_intervals = np.diff(beats)
            rhythm_regularity = np.std(beat_intervals) / np.mean(beat_intervals)
        else:
            rhythm_regularity = 1.0  # Default value
        
        # Decision rules:
        # 1. Fast tempo (>160) suggests cluttering
        # 2. High energy variance suggests stuttering (blocks and repetitions)
        # 3. Irregular rhythm suggests cluttering
        
        if tempo > 160 and rhythm_regularity > 0.5:
            return "cluttering"
        elif rms_var > 0.01:
            return "stuttering"
        else:
            # Default to stuttering as it's more common
            return "stuttering"


class HybridDisfluencyEnhancer:
    """
    Combined system that uses triage to route speech to appropriate enhancer.
    """
    def __init__(self, classifier_path=None, cluttering_model_path=None):
        """
        Initialize the hybrid enhancer.
        
        Parameters:
        -----------
        classifier_path : str, optional
            Path to pre-trained classifier model
        cluttering_model_path : str, optional
            Path to pre-trained cluttering enhancement model
        """
        self.triage = DisfluencyTriage(classifier_path)
        self.stuttering_enhancer = DisfluencyEnhancer()
        self.cluttering_enhancer = ClutteringEnhancer(cluttering_model_path)
    
    def enhance(self, audio_path: str, output_path: str, force_type=None):
        """
        Enhance speech disfluencies.
        
        Parameters:
        -----------
        audio_path : str
            Path to input audio file
        output_path : str
            Path to save enhanced audio
        force_type : str, optional
            Force specific enhancer ("stuttering" or "cluttering")
        """
        # Classify disfluency type if not forced
        if force_type is None:
            disfluency_type = self.triage.classify(audio_path)
            print(f"Detected disfluency type: {disfluency_type}")
        else:
            disfluency_type = force_type
            print(f"Using forced disfluency type: {disfluency_type}")
        
        # Route to appropriate enhancer
        if disfluency_type == "stuttering":
            print("Applying stuttering enhancement...")
            self.stuttering_enhancer.enhance(audio_path, output_path)
        else:
            print("Applying cluttering enhancement...")
            self.cluttering_enhancer.enhance(audio_path, output_path)


def train_cluttering_model(dataset_path, model_save_path, epochs=50, batch_size=16):
    """
    Train the cluttering enhancement model on a dataset of paired audio samples.
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    model_save_path : str
        Path to save trained model
    epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    """
    from torch.utils.data import Dataset, DataLoader
    import torch.optim as optim
    import glob
    
    class ClutteringDataset(Dataset):
        def __init__(self, dataset_path, n_fft=512, hop_length=128):
            self.sample_pairs = []
            self.n_fft = n_fft
            self.hop_length = hop_length
            
            # Load sample pairs (cluttered_speech, clear_speech)
            cluttered_files = glob.glob(f"{dataset_path}/cluttered/*.wav")
            
            for cluttered_file in cluttered_files:
                # Assuming clear speech files have same name in different directory
                clear_file = cluttered_file.replace("/cluttered/", "/clear/")
                if os.path.exists(clear_file):
                    self.sample_pairs.append((cluttered_file, clear_file))
            
            print(f"Loaded {len(self.sample_pairs)} sample pairs")
        
        def __len__(self):
            return len(self.sample_pairs)
        
        def __getitem__(self, idx):
            cluttered_file, clear_file = self.sample_pairs[idx]
            
            # Load audio files
            cluttered_audio, _ = librosa.load(cluttered_file, sr=16000)
            clear_audio, _ = librosa.load(clear_file, sr=16000)
            
            # Make sure both have same length
            min_len = min(len(cluttered_audio), len(clear_audio))
            cluttered_audio = cluttered_audio[:min_len]
            clear_audio = clear_audio[:min_len]
            
            # Convert to STFT domain
            cluttered_stft = librosa.stft(cluttered_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            clear_stft = librosa.stft(clear_audio, n_fft=self.n_fft, hop_length=self.hop_length)
            
            # Convert to magnitude and phase
            cluttered_mag, cluttered_phase = librosa.magphase(cluttered_stft)
            clear_mag, clear_phase = librosa.magphase(clear_stft)
            
            # Log magnitude
            cluttered_mag = np.log1p(cluttered_mag)
            clear_mag = np.log1p(clear_mag)
            
            # Phase to angle
            cluttered_phase = np.angle(cluttered_phase)
            clear_phase = np.angle(clear_phase)
            
            # Stack magnitude and phase
            cluttered_features = np.stack([cluttered_mag, cluttered_phase], axis=0)
            clear_features = np.stack([clear_mag, clear_phase], axis=0)
            
            return torch.FloatTensor(cluttered_features), torch.FloatTensor(clear_features)
    
    # Create model
    model = ClutteringEnhancerModel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Create dataset and dataloader
    dataset = ClutteringDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Loss function and optimizer
    criterion = nn.L1Loss()  # L1 loss for audio enhancement
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
    
    # Training loop
    print(f"Training on {device}...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for cluttered, clear in dataloader:
            cluttered = cluttered.to(device)
            clear = clear.to(device)
            
            # Forward pass
            outputs = model(cluttered)
            loss = criterion(outputs, clear)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}")
        
        # Adjust learning rate
        scheduler.step(epoch_loss)
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")
    
    print("Training complete!")


# Example usage:
if __name__ == "__main__":
    # Initialize the hybrid enhancer
    enhancer = HybridDisfluencyEnhancer(
        classifier_path="disfluency_classifier.joblib",
        cluttering_model_path="cluttering_model.pth"
    )
    
    # Enhance speech
    enhancer.enhance("input.wav", "enhanced.wav")
