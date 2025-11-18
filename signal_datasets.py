"""
Earthquake Signal Dataset Loader for Diffusion Transformer Training

This module implements comprehensive data loading utilities for training the signal 
diffusion transformer on broadband earthquake ground motion data (0-30 Hz). Features:

- Loading preprocessed NPZ files with STA/LTA windowed signals
- Low-pass filtering (<1 Hz) for conditioning signals
- PGA normalization and scaling
- PyTorch Dataset and DataLoader integration
- Support for data augmentation and infinite sampling
- Validation of signal quality and filtering
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torch.utils.data as data
import numpy as np
import os
import glob
from scipy import signal
import pickle
from typing import List, Dict, Optional, Tuple
import warnings

class EarthquakeSignalDataset(Dataset):
    """
    PyTorch Dataset for earthquake signals with broadband and low-frequency pairs
    
    Loads preprocessed NPZ files containing:
    - Broadband signals (0.1-30 Hz) for generation target
    - Low-frequency signals (<1 Hz) for conditioning
    - PGA values for normalization
    - Metadata for validation
    """
    
    def __init__(self, signal_paths: List[str], seq_len: int = 6000, sample_rate: float = 100.0, 
                 normalize: bool = True, pga_threshold: float = 1e-6, 
                 apply_lowpass_conditioning: bool = True, lowpass_cutoff: float = 1.0,
                 validate_signals: bool = True, transform: Optional[callable] = None):
        """
        Args:
            signal_paths: List of paths to preprocessed NPZ signal files
            seq_len: Target sequence length (default 6000 for 60s at 100Hz)
            sample_rate: Expected sampling rate in Hz  
            normalize: Whether to normalize signals by PGA
            pga_threshold: Minimum PGA threshold for valid signals
            apply_lowpass_conditioning: Whether to apply additional low-pass filtering
            lowpass_cutoff: Cutoff frequency for conditioning signal (Hz)
            validate_signals: Whether to validate loaded signals
            transform: Optional signal transformations
        """
        self.signal_paths = signal_paths
        self.seq_len = seq_len
        self.sample_rate = sample_rate
        self.normalize = normalize
        self.pga_threshold = pga_threshold
        self.apply_lowpass_conditioning = apply_lowpass_conditioning
        self.lowpass_cutoff = lowpass_cutoff
        self.validate_signals = validate_signals
        self.transform = transform
        
        # Validate dataset on initialization
        if self.validate_signals:
            self._validate_dataset()
        
        print(f"EarthquakeSignalDataset initialized:")
        print(f"  Signal files: {len(self.signal_paths)}")
        print(f"  Sequence length: {self.seq_len} samples ({self.seq_len/self.sample_rate:.1f}s)")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Normalization: {self.normalize}")
        print(f"  Low-pass conditioning: {self.apply_lowpass_conditioning} ({self.lowpass_cutoff} Hz)")
    
    def _validate_dataset(self):
        """Validate dataset files and remove invalid entries"""
        valid_paths = []
        invalid_count = 0
        
        for path in self.signal_paths:
            try:
                if os.path.exists(path) and path.endswith('.npz'):
                    # Quick validation - try to load metadata
                    data = np.load(path)
                    # Only check for broadband signal (low-pass filtering done here)
                    if 'signal_broadband' in data:
                        # Check signal quality
                        broadband = data['signal_broadband']
                        pga = np.max(np.abs(broadband))  # Calculate PGA here
                        
                        if len(broadband) >= self.seq_len // 2 and pga > self.pga_threshold:
                            valid_paths.append(path)
                        else:
                            invalid_count += 1
                    else:
                        invalid_count += 1
                else:
                    invalid_count += 1
            except Exception:
                invalid_count += 1
                continue
        
        self.signal_paths = valid_paths
        
        if invalid_count > 0:
            print(f"Dataset validation: {invalid_count} invalid files removed")
            print(f"Valid files remaining: {len(self.signal_paths)}")
        
    def __len__(self):
        return len(self.signal_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single earthquake signal sample
        
        signal_datasets.py responsibilities:
        - Load preprocessed NPZ files from bandfilter_acc.py
        - Apply low-pass filtering (<1 Hz) for conditioning signals  
        - Calculate PGA values
        - Normalize signals by PGA
        - Convert to PyTorch tensors
        
        Returns:
            Dictionary containing:
                'broadband': Broadband signal (0.1-30 Hz) for generation target
                'lowfreq': Low-frequency signal (<1 Hz) for conditioning  
                'pga_broadband': PGA of broadband signal
                'pga_lowfreq': PGA of low-frequency signal
                'metadata': Additional metadata dictionary
        """
        signal_path = self.signal_paths[idx]
        
        try:
            # Load preprocessed signal data from bandfilter_acc.py
            data = np.load(signal_path)
            
            # Extract signals - CHECK FOR PROPER PAIRS FIRST
            if 'signal_lowfreq' in data:
                # NEW: Use pre-computed proper pairs (from fixed_preprocessing.py)
                y_broad = data['signal_broadband'].copy()
                x_low = data['signal_lowfreq'].copy()
                print(f"✅ Using pre-paired signals from {os.path.basename(signal_path)}")
            else:
                # OLD: Create low-freq on-the-fly (PROBLEMATIC!)
                print(f"⚠️ Creating low-freq on-the-fly for {os.path.basename(signal_path)} - THIS IS PROBLEMATIC!")
                y_broad = data['signal_broadband'].copy()
                
                if 'signal_raw_windowed' in data:
                    raw_windowed = data['signal_raw_windowed'].copy()
                else:
                    raw_windowed = y_broad.copy()
                
                # Ensure signals are the right length
                y_broad = self._process_signal_length(y_broad)
                raw_windowed = self._process_signal_length(raw_windowed)
                
                # Apply low-pass filtering (<1 Hz) for conditioning signal
                x_low = self._apply_lowpass_filter(raw_windowed, self.lowpass_cutoff)
            
            # Handle PGA and normalization correctly for pre-processed data
            if 'normalization_pga' in data:
                # NEW: Data is already normalized! Just get original PGA values
                pga_broad = float(data['pga_broadband'])
                pga_low = float(data['pga_lowfreq'])
                norm_pga = float(data['normalization_pga'])
                print(f"   ✅ Data already normalized by PGA: {norm_pga:.6f}")
                print(f"   Signal ranges after loading: y_broad=[{y_broad.min():.3f}, {y_broad.max():.3f}], x_low=[{x_low.min():.3f}, {x_low.max():.3f}]")
                
                # Signals should already be in [-1, 1] range
                if self.normalize:
                    # Already normalized, but store normalized PGA values
                    pga_broad_norm = pga_broad / norm_pga
                    pga_low_norm = pga_low / norm_pga
                else:
                    # If user wants unnormalized, multiply back
                    y_broad = y_broad * norm_pga
                    x_low = x_low * norm_pga
                    pga_broad_norm = pga_broad
                    pga_low_norm = pga_low
            else:
                # OLD: Data not pre-normalized, calculate PGA and normalize here
                pga_broad = np.max(np.abs(y_broad))
                pga_low = np.max(np.abs(x_low))
                
                if self.normalize:
                    norm_pga = pga_broad  # Use broadband PGA
                    if norm_pga > self.pga_threshold:
                        y_broad = y_broad / norm_pga
                        x_low = x_low / norm_pga
                        pga_broad_norm = 1.0  # Normalized to 1
                        pga_low_norm = pga_low / norm_pga
                else:
                    pga_broad_norm = pga_broad
                    pga_low_norm = pga_low
            
            # Validate signal quality
            if self.validate_signals:
                if not self._is_valid_signal(y_broad, max(abs(y_broad.min()), abs(y_broad.max()))):
                    warnings.warn(f"Invalid signal in {signal_path}, using fallback")
                    return self._get_fallback_sample()
            
            # Apply transforms if provided
            if self.transform:
                y_broad = self.transform(y_broad)
                x_low = self.transform(x_low)
            
            # Create metadata dictionary
            metadata = {
                'file_path': signal_path,
                'original_file': str(data.get('original_file', 'unknown')),
                'sample_rate': float(data.get('sample_rate', self.sample_rate)),
                'duration': float(data.get('duration', len(y_broad) / self.sample_rate)),
                'trigger_idx': int(data.get('trigger_idx', -1)),
                'component': str(data.get('component', 'unknown')),
                'augmentation_type': str(data.get('augmentation_type', 'original')),
                'augmentation_applied': [],  # FIX: Skip corrupted augmentation_applied field
                'rotation_angle': float(data.get('rotation_angle', 0.0))
            }
            
            # Convert to tensors
            y_broad = torch.tensor(y_broad, dtype=torch.float32)
            x_low = torch.tensor(x_low, dtype=torch.float32)
            pga_broad_tensor = torch.tensor(pga_broad_norm, dtype=torch.float32)
            pga_low_tensor = torch.tensor(pga_low_norm, dtype=torch.float32)
            
            return {
                'broadband': y_broad,      # 0.1-30 Hz (generation target)
                'lowfreq': x_low,          # <1 Hz (conditioning signal)
                'pga_broadband': pga_broad_tensor,
                'pga_lowfreq': pga_low_tensor,
                'metadata': metadata
            }
            
        except Exception as e:
            warnings.warn(f"Error loading {signal_path}: {e}")
            return self._get_fallback_sample()
    
    def _is_valid_signal(self, signal: np.ndarray, pga: float) -> bool:
        """Validate signal quality"""
        if len(signal) < self.seq_len // 2:
            return False
        if pga < self.pga_threshold:
            return False
        if np.any(np.isnan(signal)) or np.any(np.isinf(signal)):
            return False
        return True
    
    def _get_fallback_sample(self) -> Dict[str, torch.Tensor]:
        """Return a fallback sample for corrupted data"""
        # Create synthetic noise signal as fallback
        y_broad = np.random.randn(self.seq_len) * 1e-4
        x_low = np.random.randn(self.seq_len) * 1e-5
        
        return {
            'broadband': torch.tensor(y_broad, dtype=torch.float32),
            'lowfreq': torch.tensor(x_low, dtype=torch.float32),
            'pga_broadband': torch.tensor(1e-4, dtype=torch.float32),
            'pga_lowfreq': torch.tensor(1e-5, dtype=torch.float32),
            'metadata': {
                'file_path': 'fallback',
                'original_file': 'fallback',
                'sample_rate': self.sample_rate,
                'duration': self.seq_len / self.sample_rate,
                'trigger_idx': -1,
                'component': 'fallback',
                'augmentation_type': 'fallback'
            }
        }
    
    def _process_signal_length(self, signal: np.ndarray) -> np.ndarray:
        """
        Ensure signal has the correct length for training
        
        Args:
            signal: Input signal array
            
        Returns:
            processed_signal: Signal of target length
        """
        if len(signal) >= self.seq_len:
            # Randomly crop to desired length for data augmentation
            max_start = len(signal) - self.seq_len
            start_idx = np.random.randint(0, max_start + 1)
            return signal[start_idx:start_idx + self.seq_len].copy()
        else:
            # Pad with zeros if too short (maintain 60s duration)
            padding = self.seq_len - len(signal)
            return np.pad(signal, (0, padding), mode='constant', constant_values=0)
    
    def _apply_bandpass_filter(self, data, lowcut, highcut, order=4):
        """Apply Butterworth band-pass filter"""
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        # Ensure frequencies are valid
        low = max(0.01, min(low, 0.99))
        high = max(low + 0.01, min(high, 0.99))
        
        try:
            b, a = signal.butter(order, [low, high], btype='band')
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data
        except:
            # If filtering fails, return original data
            return data
    
    def _apply_lowpass_filter(self, data, cutoff, order=4):
        """Apply Butterworth low-pass filter"""
        nyquist = 0.5 * self.sample_rate
        normalized_cutoff = cutoff / nyquist
        
        # Ensure frequency is valid
        normalized_cutoff = min(normalized_cutoff, 0.99)
        
        try:
            b, a = signal.butter(order, normalized_cutoff, btype='low')
            filtered_data = signal.filtfilt(b, a, data)
            return filtered_data
        except:
            # If filtering fails, return original data
            return data

class InfiniteSignalSampler(data.sampler.Sampler):
    """Infinite sampler for continuous training"""
    def __init__(self, data_source):
        self.num_samples = len(data_source)

    def __iter__(self):
        while True:
            yield from torch.randperm(self.num_samples).tolist()

    def __len__(self):
        return 2 ** 31

def list_signal_files(data_dir, extensions=('.npz', '.npy')):
    """List all signal files in directory and subdirectories"""
    signal_files = []
    for ext in extensions:
        pattern = os.path.join(data_dir, '**', f'*{ext}')
        signal_files.extend(glob.glob(pattern, recursive=True))
    return sorted(signal_files)

def create_signal_loader(data_dir: str, seq_len: int = 6000, batch_size: int = 32, 
                        sample_rate: float = 100.0, normalize: bool = True, 
                        pga_threshold: float = 1e-6, apply_lowpass_conditioning: bool = True,
                        lowpass_cutoff: float = 1.0, validate_signals: bool = True,
                        num_workers: int = 4, infinite: bool = True, 
                        train_split: float = 0.8) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create data loaders for earthquake signal training
    
    Args:
        data_dir: Directory containing preprocessed NPZ signal files
        seq_len: Target sequence length (default 6000 for 60s at 100Hz)
        batch_size: Batch size for training
        sample_rate: Expected sampling rate in Hz
        normalize: Whether to normalize signals by PGA
        pga_threshold: Minimum PGA threshold for valid signals
        apply_lowpass_conditioning: Whether to apply additional low-pass filtering
        lowpass_cutoff: Cutoff frequency for conditioning (Hz)
        validate_signals: Whether to validate signal quality
        num_workers: Number of worker processes
        infinite: Whether to use infinite sampling for training
        train_split: Fraction of data for training (rest for validation)
    
    Returns:
        train_loader: Training data loader
        val_loader: Validation data loader (None if train_split >= 1.0)
    """
    # Find all preprocessed signal files
    signal_files = list_signal_files(data_dir, extensions=('.npz',))
    
    if len(signal_files) == 0:
        raise ValueError(f"No NPZ signal files found in {data_dir}")
    
    print(f"Found {len(signal_files)} preprocessed signal files")
    
    # Split into train/validation
    np.random.shuffle(signal_files)
    n_train = int(len(signal_files) * train_split)
    
    train_files = signal_files[:n_train]
    val_files = signal_files[n_train:] if train_split < 1.0 else []
    
    print(f"Train files: {len(train_files)}, Validation files: {len(val_files)}")
    
    # Create training dataset
    train_dataset = EarthquakeSignalDataset(
        train_files,
        seq_len=seq_len,
        sample_rate=sample_rate,
        normalize=normalize,
        pga_threshold=pga_threshold,
        apply_lowpass_conditioning=apply_lowpass_conditioning,
        lowpass_cutoff=lowpass_cutoff,
        validate_signals=validate_signals
    )
    
    # Create training loader
    if infinite:
        sampler = InfiniteSignalSampler(train_dataset)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )
    
    # Create validation loader if needed
    val_loader = None
    if val_files:
        val_dataset = EarthquakeSignalDataset(
            val_files,
            seq_len=seq_len,
            sample_rate=sample_rate,
            normalize=normalize,
            pga_threshold=pga_threshold,
            apply_lowpass_conditioning=apply_lowpass_conditioning,
            lowpass_cutoff=lowpass_cutoff,
            validate_signals=validate_signals
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(num_workers, 2),
            pin_memory=True,
            drop_last=False
        )
    
    return train_loader, val_loader

def preprocess_at2_files_for_training(input_dir: str, output_dir: str, 
                                     apply_augmentation: bool = True, 
                                     num_augmentations: int = 3) -> None:
    """
    Preprocess AT2 files using the comprehensive bandfilter_acc pipeline
    
    This function calls the preprocessing pipeline from bandfilter_acc.py which includes:
    - STA/LTA and Z-detector event detection
    - 60-second time windowing with zero-padding
    - Data augmentation (time shifts, horizontal rotation)
    - Band-pass filtering (0.1-30 Hz) and low-pass filtering (<1 Hz)
    - NPZ format output with metadata
    
    Args:
        input_dir: Directory containing AT2 files
        output_dir: Directory to save processed NPZ files
        apply_augmentation: Whether to apply data augmentation
        num_augmentations: Number of augmented versions per signal
    """
    try:
        from data_prep_acc.bandfilter_acc import process_all_for_training
        
        print("Using comprehensive preprocessing pipeline from bandfilter_acc.py")
        process_all_for_training(
            input_dir=input_dir,
            output_dir=output_dir,
            apply_augmentation=apply_augmentation,
            num_augmentations=num_augmentations
        )
        
    except ImportError:
        print("Warning: Could not import bandfilter_acc module. Using basic processing.")
        _basic_at2_processing(input_dir, output_dir)

def _basic_at2_processing(input_dir: str, output_dir: str) -> None:
    """Basic AT2 processing fallback"""
    from data_prep_acc.bandfilter_acc import read_at2, butterworth_bandpass_filter, butterworth_lowpass_filter
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all AT2 files
    at2_files = glob.glob(os.path.join(input_dir, '**', '*.AT2'), recursive=True)
    
    print(f"Basic processing of {len(at2_files)} AT2 files...")
    
    for i, at2_file in enumerate(at2_files):
        try:
            # Read AT2 file
            accel, dt, npts = read_at2(at2_file)
            fs = 1.0 / dt
            
            # Apply band-pass filter (0.1-30 Hz)
            broadband, _, _ = butterworth_bandpass_filter(accel, 0.1, 30.0, fs, 4)
            
            # Apply low-pass filter (<1 Hz) for conditioning
            lowfreq, _, _ = butterworth_lowpass_filter(accel, 1.0, fs, 4)
            
            # Ensure 60-second duration (6000 samples at 100 Hz)
            target_samples = 6000
            if len(broadband) >= target_samples:
                broadband = broadband[:target_samples]
                lowfreq = lowfreq[:target_samples]
            else:
                # Zero-pad if too short
                pad_size = target_samples - len(broadband)
                broadband = np.pad(broadband, (0, pad_size), mode='constant')
                lowfreq = np.pad(lowfreq, (0, pad_size), mode='constant')
            
            # Compute PGAs
            pga_broadband = np.max(np.abs(broadband))
            pga_lowfreq = np.max(np.abs(lowfreq))
            
            # Create output filename
            base_name = os.path.splitext(os.path.basename(at2_file))[0]
            output_path = os.path.join(output_dir, f"{base_name}_basic.npz")
            
            # Save as NPZ
            np.savez(output_path,
                    signal_broadband=broadband,
                    signal_lowfreq=lowfreq,
                    pga_broadband=pga_broadband,
                    pga_lowfreq=pga_lowfreq,
                    sample_rate=100.0,
                    dt=0.01,
                    duration=60.0,
                    trigger_idx=-1,
                    original_file=at2_file,
                    augmentation_type='basic')
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(at2_files)} files")
                
        except Exception as e:
            print(f"Error processing {at2_file}: {e}")
            continue
    
    print(f"Basic processing complete! Saved files to {output_dir}")

# Data augmentation transforms
class SignalTransforms:
    """Collection of signal augmentation transforms"""
    
    @staticmethod
    def add_noise(signal, noise_level=0.01):
        """Add Gaussian noise to signal"""
        noise = torch.randn_like(signal) * noise_level
        return signal + noise
    
    @staticmethod
    def time_shift(signal, max_shift=100):
        """Randomly shift signal in time"""
        shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
        if shift > 0:
            return torch.cat([torch.zeros(shift), signal[:-shift]])
        elif shift < 0:
            return torch.cat([signal[-shift:], torch.zeros(-shift)])
        else:
            return signal
    
    @staticmethod
    def amplitude_scale(signal, scale_range=(0.8, 1.2)):
        """Randomly scale signal amplitude"""
        scale = torch.uniform(*scale_range)
        return signal * scale

def validate_dataset_integrity(data_dir: str) -> Dict[str, int]:
    """
    Validate the integrity of a processed earthquake signal dataset
    
    Args:
        data_dir: Directory containing NPZ files
        
    Returns:
        stats: Dictionary with dataset statistics
    """
    signal_files = list_signal_files(data_dir, extensions=('.npz',))
    
    stats = {
        'total_files': len(signal_files),
        'valid_files': 0,
        'corrupted_files': 0,
        'short_signals': 0,
        'low_pga_signals': 0,
        'augmented_files': 0,
        'original_files': 0
    }
    
    pga_values = []
    durations = []
    
    for file_path in signal_files:
        try:
            data = np.load(file_path)
            
            # Check required fields
            if 'signal_broadband' not in data or 'signal_lowfreq' not in data:
                stats['corrupted_files'] += 1
                continue
                
            broadband = data['signal_broadband']
            pga = float(data.get('pga_broadband', np.max(np.abs(broadband))))
            duration = float(data.get('duration', len(broadband) * 0.01))
            aug_type = str(data.get('augmentation_type', 'unknown'))
            
            # Collect statistics
            pga_values.append(pga)
            durations.append(duration)
            
            if duration < 30.0:  # Less than 30 seconds
                stats['short_signals'] += 1
                
            if pga < 1e-6:  # Very low PGA
                stats['low_pga_signals'] += 1
                
            if 'original' in aug_type:
                stats['original_files'] += 1
            else:
                stats['augmented_files'] += 1
                
            stats['valid_files'] += 1
            
        except Exception:
            stats['corrupted_files'] += 1
            continue
    
    # Summary statistics
    if pga_values:
        stats['pga_mean'] = np.mean(pga_values)
        stats['pga_std'] = np.std(pga_values)
        stats['pga_min'] = np.min(pga_values)
        stats['pga_max'] = np.max(pga_values)
        
    if durations:
        stats['duration_mean'] = np.mean(durations)
        stats['duration_std'] = np.std(durations)
        
    return stats

def print_dataset_summary(stats: Dict[str, int]):
    """Print a formatted summary of dataset statistics"""
    print(f"\n{'='*60}")
    print(f"EARTHQUAKE SIGNAL DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total files found: {stats['total_files']}")
    print(f"Valid files: {stats['valid_files']}")
    print(f"Corrupted files: {stats['corrupted_files']}")
    print(f"Short signals (<30s): {stats['short_signals']}")
    print(f"Low PGA signals (<1e-6): {stats['low_pga_signals']}")
    print(f"Original signals: {stats['original_files']}")
    print(f"Augmented signals: {stats['augmented_files']}")
    
    if 'pga_mean' in stats:
        print(f"\nPGA Statistics:")
        print(f"  Mean: {stats['pga_mean']:.2e} g")
        print(f"  Std:  {stats['pga_std']:.2e} g")  
        print(f"  Min:  {stats['pga_min']:.2e} g")
        print(f"  Max:  {stats['pga_max']:.2e} g")
        
    if 'duration_mean' in stats:
        print(f"\nDuration Statistics:")
        print(f"  Mean: {stats['duration_mean']:.1f} s")
        print(f"  Std:  {stats['duration_std']:.1f} s")
    
    print(f"{'='*60}")

# Test and validation functions
if __name__ == "__main__":
    print("Testing Earthquake Signal Dataset...")
    
    # Create dummy data for testing with proper format
    test_dir = "test_signals"
    os.makedirs(test_dir, exist_ok=True)
    
    # Generate some test signals in the expected NPZ format
    for i in range(10):
        # Create synthetic earthquake-like signal (60 seconds at 100 Hz)
        t = np.linspace(0, 60, 6000)
        
        # Synthetic broadband signal with realistic earthquake characteristics
        broadband = (0.1 * np.sin(2 * np.pi * 0.5 * t) * np.exp(-0.05 * t) +
                    0.05 * np.sin(2 * np.pi * 2 * t) * np.exp(-0.02 * t) +
                    0.02 * np.sin(2 * np.pi * 10 * t) * np.exp(-0.1 * t) +
                    0.005 * np.random.randn(len(t)))
        
        # Synthetic low-frequency conditioning signal
        lowfreq = 0.05 * np.sin(2 * np.pi * 0.3 * t) * np.exp(-0.03 * t)
        
        pga_broad = np.max(np.abs(broadband))
        pga_low = np.max(np.abs(lowfreq))
        
        # Save as NPZ with proper format
        np.savez(os.path.join(test_dir, f"test_signal_{i:03d}.npz"),
                signal_broadband=broadband,
                signal_lowfreq=lowfreq,
                pga_broadband=pga_broad,
                pga_lowfreq=pga_low,
                sample_rate=100.0,
                dt=0.01,
                duration=60.0,
                trigger_idx=500,
                original_file=f"test_{i}.AT2",
                augmentation_type='original')
    
    # Test dataset loading
    try:
        train_loader, val_loader = create_signal_loader(
            test_dir, 
            seq_len=6000, 
            batch_size=4,
            infinite=False
        )
        
        # Get a batch from training loader
        batch = next(iter(train_loader))
        
        print(f"Batch keys: {list(batch.keys())}")
        print(f"Broadband shape: {batch['broadband'].shape}")
        print(f"Low-freq shape: {batch['lowfreq'].shape}")
        print(f"PGA broadband range: {batch['pga_broadband'].min():.2e} - {batch['pga_broadband'].max():.2e}")
        print(f"PGA low-freq range: {batch['pga_lowfreq'].min():.2e} - {batch['pga_lowfreq'].max():.2e}")
        print(f"Metadata keys: {list(batch['metadata'].keys()) if 'metadata' in batch else 'No metadata'}")
        
        # Test dataset validation
        stats = validate_dataset_integrity(test_dir)
        print_dataset_summary(stats)
        
        print("\n✅ Dataset test successful!")
        
    except Exception as e:
        print(f"❌ Error in dataset test: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up test data
        import shutil
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
            print("Test data cleaned up.")
