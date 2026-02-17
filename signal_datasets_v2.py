"""Earthquake Signal Dataset V2 with Variable-Length Packing - Simplified

New features:
- Variable-length signal support (no zero-padding)
- Sequence packing with [SEP] tokens
- Batch collation with block-diagonal attention masks
- Works with RAW signals in ORIGINAL scale (no normalization, no log-space)
- RoPE-compatible position IDs
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from typing import List, Dict, Optional, Tuple
from scipy import signal
from scipy.signal import hilbert

from packed_attention import PackedSequenceManager


class EarthquakeSignalDatasetV2(Dataset):
    """
    PyTorch Dataset for variable-length earthquake signals - Simplified.
    
    Loads preprocessed NPZ files and returns:
    - Raw signals in ORIGINAL scale (no normalization, no log-space)
    - No padding (variable lengths preserved)
    - Lowpass signal extraction for ControlNet
    """
    
    def __init__(
        self,
        signal_paths: List[str],
        sample_rate: float = 100.0,
        min_length: int = 5000,
        max_length: int = 50000,
        pga_threshold: float = 1e-6,
        lowpass_cutoff: float = 1.0,
        log_epsilon: float = 1e-6,
        validate_signals: bool = True
    ):
        """
        Args:
            signal_paths: List of paths to preprocessed NPZ files
            sample_rate: Sampling rate in Hz
            min_length: Minimum signal length
            max_length: Maximum signal length
            pga_threshold: Minimum PGA for valid signals
            lowpass_cutoff: Cutoff frequency for lowpass conditioning
            log_epsilon: Epsilon for log-space transformation
            validate_signals: Whether to validate dataset
        """
        self.signal_paths = signal_paths
        self.sample_rate = sample_rate
        self.min_length = min_length
        self.max_length = max_length
        self.pga_threshold = pga_threshold
        self.lowpass_cutoff = lowpass_cutoff
        self.log_epsilon = log_epsilon  # Kept for interface compatibility but NOT USED
        
        # Validate dataset
        if validate_signals:
            self._validate_dataset()
        
        print(f"EarthquakeSignalDatasetV2 initialized (ORIGINAL scale - no log-space):")
        print(f"  Signal files: {len(self.signal_paths)}")
        print(f"  Length range: [{self.min_length}, {self.max_length}] samples")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Lowpass cutoff: {self.lowpass_cutoff} Hz")
    
    def _validate_dataset(self):
        """Validate and filter signal files"""
        valid_paths = []
        invalid_count = 0
        
        for path in self.signal_paths:
            try:
                if not os.path.exists(path) or not path.endswith('.npz'):
                    invalid_count += 1
                    continue
                
                data = np.load(path)
                
                # Check for broadband signal (support both key formats)
                if 'signal_broadband' in data:
                    broadband = data['signal_broadband']
                elif 'signal_normalized' in data:
                    broadband = data['signal_normalized']
                else:
                    invalid_count += 1
                    continue
                
                signal_len = len(broadband)
                
                # Get PGA (for validation, use stored or computed value)
                if 'pga_broadband' in data:
                    pga = float(data['pga_broadband'])
                else:
                    pga = np.max(np.abs(broadband))
                
                # Check length and quality
                if (self.min_length <= signal_len <= self.max_length and 
                    pga > self.pga_threshold):
                    valid_paths.append(path)
                else:
                    invalid_count += 1
            
            except Exception:
                invalid_count += 1
                continue
        
        self.signal_paths = valid_paths
        
        if invalid_count > 0:
            print(f"  Validation: {invalid_count} invalid files removed")
            print(f"  Valid files: {len(self.signal_paths)}")
    
    def _apply_lowpass_filter(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply lowpass filter for conditioning"""
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = self.lowpass_cutoff / nyquist
        
        # Butterworth lowpass filter
        b, a = signal.butter(4, normalized_cutoff, btype='low')
        lowpass = signal.filtfilt(b, a, signal_data)
        
        return lowpass
    
    def _compute_envelope(self, signal_data: np.ndarray) -> np.ndarray:
        """Compute amplitude envelope using Hilbert transform"""
        try:
            # Compute analytic signal using Hilbert transform
            analytic_signal = hilbert(signal_data)
            # Envelope is the magnitude of the analytic signal
            envelope = np.abs(analytic_signal)
        except Exception as e:
            # Fallback: use absolute value (crude envelope)
            print(f"Warning: Hilbert transform failed, using absolute value: {e}")
            envelope = np.abs(signal_data)
        return envelope
    
    def __len__(self):
        return len(self.signal_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load and process a single signal.
        
        Returns:
            Dictionary with:
                'broadband': Broadband signal in ORIGINAL scale [seq_len]
                'lowpass': Lowpass conditioning signal in ORIGINAL scale [seq_len]
                'metadata': Seismic metadata [3] (M, Vs30, HypD)
                'length': Actual signal length (scalar)
                'pga': Original PGA value (scalar)
        """
        signal_path = self.signal_paths[idx]
        
        try:
            data = np.load(signal_path)
            
            # Extract broadband signal (try both old and new key formats)
            if 'signal_broadband' in data:
                broadband_normalized = data['signal_broadband'].copy()
            elif 'signal_normalized' in data:
                broadband_normalized = data['signal_normalized'].copy()
            else:
                raise KeyError("No broadband signal found in NPZ file")
            
            signal_len = len(broadband_normalized)
            
            # Clip to max length if needed
            if signal_len > self.max_length:
                broadband_normalized = broadband_normalized[:self.max_length]
                signal_len = self.max_length
            
            # FIX BUG #1: Signals are saved RAW (not normalized), so NO denormalization needed!
            # The preprocessing now saves signals in original physical scale.
            # PGA is saved only for reference, not for denormalization.
            if 'pga_broadband' in data:
                pga = float(data['pga_broadband'])  # Saved for metadata only
            else:
                pga = np.max(np.abs(broadband_normalized))
            
            # Use signal directly - it's already in correct RAW scale
            broadband = broadband_normalized
            
            # Compute envelope from broadband signal
            envelope = self._compute_envelope(broadband)
            
            # Extract or create lowpass signal
            if 'signal_lowfreq' in data:
                # FIX BUG #3: Lowpass is also saved RAW, use directly
                lowpass = data['signal_lowfreq'].copy()
                if len(lowpass) > self.max_length:
                    lowpass = lowpass[:self.max_length]
                # PGA saved for reference only
                if 'pga_lowfreq' in data:
                    pga_lowfreq = float(data['pga_lowfreq'])
            elif 'lowfreq_normalized' in data:
                # Legacy support: old format (actually already in correct scale)
                lowpass = data['lowfreq_normalized'].copy()
                if len(lowpass) > self.max_length:
                    lowpass = lowpass[:self.max_length]
            else:
                # Generate lowpass by filtering the RAW broadband signal
                lowpass = self._apply_lowpass_filter(broadband)
            
            # Extract metadata
            if 'magnitude' in data and 'vs30' in data and 'hypocenter_distance' in data:
                magnitude = float(data['magnitude'])
                vs30 = float(data['vs30'])
                hyp_dist = float(data['hypocenter_distance'])
            else:
                # Default values if not available
                magnitude = 6.0
                vs30 = 300.0
                hyp_dist = 50.0
            
            # Convert to tensors - keep in ORIGINAL scale (no log-space, already denormalized)
            broadband_tensor = torch.from_numpy(broadband).float()
            lowpass_tensor = torch.from_numpy(lowpass).float()
            envelope_tensor = torch.from_numpy(envelope).float()
            
            # Metadata tensor
            metadata = torch.tensor([magnitude, vs30, hyp_dist], dtype=torch.float32)
            
            return {
                'broadband': broadband_tensor,  # Original scale
                'lowpass': lowpass_tensor,  # Original scale
                'envelope': envelope_tensor,  # Original scale
                'metadata': metadata,
                'length': signal_len,
                'pga': pga
            }
        
        except Exception as e:
            print(f"Error loading {signal_path}: {e}")
            # Return dummy data
            dummy_len = self.min_length
            return {
                'broadband': torch.zeros(dummy_len),
                'lowpass': torch.zeros(dummy_len),
                'envelope': torch.zeros(dummy_len),
                'metadata': torch.tensor([6.0, 300.0, 50.0]),
                'length': dummy_len,
                'pga': 0.0
            }


def collate_packed_sequences(
    batch: List[Dict[str, torch.Tensor]],
    pack_size: int = 3,
    max_length: int = 50000,
    pad_value: float = 0.0
) -> Dict[str, torch.Tensor]:
    """
    Collate function for packing variable-length sequences.
    
    Args:
        batch: List of samples from dataset
        pack_size: Number of sequences to pack together
        max_length: Maximum total length after packing
        pad_value: Padding value
    
    Returns:
        Dictionary with packed batches:
            'broadband': Packed broadband signals [pack_size, max_length]
            'lowpass': Packed lowpass signals [pack_size, max_length]
            'envelope': Packed envelope signals [pack_size, max_length]
            'metadata': Metadata for each sequence [pack_size, 3]
            'position_ids': Position IDs with resets [pack_size, max_length]
            'boundaries': List of (start, end) tuples
            'lengths': Actual lengths [pack_size]
            'pgas': Original PGA values [pack_size]
    """
    # Sort by length for efficient packing
    batch = sorted(batch, key=lambda x: x['length'], reverse=True)
    
    # Take pack_size samples
    if len(batch) > pack_size:
        batch = batch[:pack_size]
    
    # Extract sequences
    broadband_seqs = [item['broadband'] for item in batch]
    lowpass_seqs = [item['lowpass'] for item in batch]
    envelope_seqs = [item['envelope'] for item in batch]
    metadata_list = [item['metadata'] for item in batch]
    lengths = [item['length'] for item in batch]
    pgas = [item['pga'] for item in batch]
    
    # Pack sequences
    manager = PackedSequenceManager()
    
    packed_broadband, lengths_tensor, boundaries = manager.pack_sequences(
        broadband_seqs, max_length=max_length, pad_value=pad_value, add_sep=True
    )
    
    packed_lowpass, _, _ = manager.pack_sequences(
        lowpass_seqs, max_length=max_length, pad_value=pad_value, add_sep=True
    )
    
    packed_envelope, _, _ = manager.pack_sequences(
        envelope_seqs, max_length=max_length, pad_value=pad_value, add_sep=True
    )
    
    # Stack metadata
    metadata_tensor = torch.stack(metadata_list, dim=0)
    
    # Use actual packed length instead of max_length
    actual_length = boundaries[-1][1]  # Last boundary end position
    
    # Don't materialize the full mask - just store boundaries
    # The model will handle masking during attention computation
    # This saves enormous amounts of memory for long sequences
    
    # Create position IDs - only for actual content
    position_ids_content = manager.create_position_ids(boundaries, actual_length, device=packed_broadband.device)
    
    # Pad position IDs to max_length (padding positions get -1)
    if actual_length < max_length:
        pad_size = max_length - actual_length
        position_ids = torch.full((pack_size, max_length), -1, dtype=torch.long, device=packed_broadband.device)
        position_ids[:, :actual_length] = position_ids_content.unsqueeze(0).expand(pack_size, -1)
    else:
        position_ids = position_ids_content.unsqueeze(0).expand(pack_size, -1)
    
    # Convert PGAs to tensor
    pga_tensor = torch.tensor(pgas, dtype=torch.float32)
    
    return {
        'broadband': packed_broadband,
        'lowpass': packed_lowpass,
        'envelope': packed_envelope,
        'metadata': metadata_tensor,
        'position_ids': position_ids,
        'boundaries': boundaries,  # Store boundaries instead of materialized mask
        'actual_length': actual_length,  # Store actual length for attention
        'lengths': lengths_tensor,
        'pgas': pga_tensor
    }


def create_dataloader_v2(
    signal_paths: List[str],
    batch_size: int = 8,
    pack_size: int = 3,
    max_length: int = 50000,
    num_workers: int = 4,
    shuffle: bool = True,
    **dataset_kwargs
) -> DataLoader:
    """
    Create DataLoader for variable-length packed sequences.
    
    Args:
        signal_paths: List of signal file paths
        batch_size: Number of packed batches
        pack_size: Number of sequences per pack
        max_length: Maximum length after packing
        num_workers: Number of worker processes
        shuffle: Whether to shuffle data
        **dataset_kwargs: Additional arguments for dataset
    
    Returns:
        DataLoader with packed collation
    """
    dataset = EarthquakeSignalDatasetV2(signal_paths, **dataset_kwargs)
    
    from functools import partial
    collate_fn = partial(
        collate_packed_sequences,
        pack_size=pack_size,
        max_length=max_length
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return dataloader


if __name__ == "__main__":
    # Test dataset
    print("Testing EarthquakeSignalDatasetV2...")
    
    # Get signal paths (assuming data_prep_acc structure)
    data_dir = "data_prep_acc/processed_dynamic"
    if os.path.exists(data_dir):
        signal_paths = [
            os.path.join(data_dir, f) 
            for f in os.listdir(data_dir) 
            if f.endswith('.npz')
        ][:10]  # Test with first 10
        
        if len(signal_paths) > 0:
            # Create dataset
            dataset = EarthquakeSignalDatasetV2(
                signal_paths,
                sample_rate=100.0,
                min_length=5000,
                max_length=50000,
                validate_signals=True
            )
            
            print(f"Dataset size: {len(dataset)}")
            
            # Test single sample
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            print(f"Broadband shape: {sample['broadband'].shape}")
            print(f"Lowpass shape: {sample['lowpass'].shape}")
            print(f"Metadata: {sample['metadata']}")
            print(f"Length: {sample['length']}")
            
            # Test dataloader - NO PACKING for testing small dataset
            dataloader = create_dataloader_v2(
                signal_paths,
                batch_size=4,  # Process 4 individual sequences per batch
                pack_size=1,  # No packing - each sequence separate
                max_length=35000,  # Single sequence max length
                num_workers=0,
                shuffle=False
            )
            
            batch = next(iter(dataloader))
            print(f"\nPacked batch keys: {batch.keys()}")
            print(f"Packed broadband shape: {batch['broadband'].shape}")
            print(f"Position IDs shape: {batch['position_ids'].shape}")
            print(f"Boundaries: {batch['boundaries']}")
            print(f"Actual length: {batch['actual_length']}")
            print(f"Lengths: {batch['lengths']}")
            
            print("\n✓ Dataset test passed!")
            print("Note: Attention mask is computed on-the-fly during forward pass to save memory")
        else:
            print("No signal files found for testing")
    else:
        print(f"Data directory not found: {data_dir}")
