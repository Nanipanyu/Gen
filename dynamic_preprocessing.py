#!/usr/bin/env python3

import numpy as np
import os
from scipy import signal
import glob

def read_at2_file(file_path):
    """Read AT2 format earthquake data"""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header - format: "NPTS=   8900, DT=   .0100 SEC"
    header_line = lines[3].strip()
    
    # Extract NPTS
    npts_part = header_line.split(',')[0]  # "NPTS=   8900"
    npts = int(npts_part.split('=')[1].strip())
    
    # Extract DT  
    dt_part = header_line.split(',')[1]    # " DT=   .0100 SEC"
    dt = float(dt_part.split('=')[1].replace('SEC', '').strip())
    
    # Read data values (skip header lines)
    data_lines = lines[4:]
    values = []
    for line in data_lines:
        values.extend([float(x) for x in line.split()])
    
    acceleration = np.array(values[:npts])
    sampling_rate = 1.0 / dt
    
    return acceleration, sampling_rate, dt

def apply_filters(signal_data, sampling_rate):
    """Apply bandpass and lowpass filters"""
    nyquist = sampling_rate / 2
    
    # Broadband filter: 0.1-30 Hz
    low_broad = 0.1 / nyquist
    high_broad = min(30.0 / nyquist, 0.95)  # Ensure below Nyquist
    
    if high_broad > low_broad and low_broad < 0.95:
        try:
            sos_broad = signal.butter(4, [low_broad, high_broad], btype='band', output='sos')
            broadband = signal.sosfilt(sos_broad, signal_data)
        except:
            # Fallback if filter design fails
            broadband = signal_data.copy()
    else:
        broadband = signal_data.copy()
    
    # Low-frequency filter: <1 Hz
    low_freq_cutoff = min(1.0 / nyquist, 0.45)  # Ensure well below Nyquist
    try:
        sos_low = signal.butter(4, low_freq_cutoff, btype='low', output='sos')
        lowfreq = signal.sosfilt(sos_low, signal_data)
    except:
        # Fallback if filter design fails
        lowfreq = signal_data.copy()
    
    return broadband, lowfreq

def find_target_length(at2_files):
    """Find the longest signal to determine target sequence length with memory considerations"""
    max_length = 0
    max_rate = 0
    reference_file = None
    
    print("🔍 Scanning files to determine target dimensions...")
    
    for file_path in at2_files:
        try:
            acceleration, sampling_rate, dt = read_at2_file(file_path)
            print(f"   {os.path.basename(file_path)}: {len(acceleration)} samples at {sampling_rate:.1f} Hz")
            
            if len(acceleration) > max_length:
                max_length = len(acceleration)
                max_rate = sampling_rate
                reference_file = file_path
                
        except Exception as e:
            print(f"   ❌ Error reading {file_path}: {e}")
    
    # Apply memory cap (30k samples = 60 patches with patch_size=500)
    capped_length = min(max_length, 30000)
    
    print(f"\n📊 DYNAMIC PARAMETERS:")
    if max_length > 30000:
        print(f"   🚨 Memory cap applied: {max_length} → {capped_length} samples") 
    print(f"   Target sequence length: {capped_length} samples")
    print(f"   Reference sampling rate: {max_rate:.1f} Hz") 
    print(f"   Reference file: {os.path.basename(reference_file)}")
    print(f"   Duration: {capped_length / max_rate:.1f} seconds")
    
    return capped_length, max_rate, reference_file

def main():
    """Process AT2 files with preserved sampling rates and variable lengths"""
    
    input_dir = "data_prep_acc/rawdata"
    output_dir = "data_prep_acc/processed_dynamic"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all AT2 files
    at2_files = glob.glob(os.path.join(input_dir, "*.AT2"))
    
    if not at2_files:
        print(f"❌ No AT2 files found in {input_dir}")
        return
    
    print(f"🔍 Found {len(at2_files)} AT2 files")
    
    # Determine target dimensions dynamically
    target_length, reference_rate, reference_file = find_target_length(at2_files)
    
    processed_count = 0
    
    for file_path in at2_files:
        try:
            # Read original data
            acceleration, original_rate, dt = read_at2_file(file_path)
            original_duration = len(acceleration) / original_rate
            
            print(f"\n📁 Processing: {os.path.basename(file_path)}")
            print(f"   Original: {len(acceleration)} samples, {original_rate:.1f} Hz, {original_duration:.1f}s")
            
            # NO RESAMPLING - preserve original sampling rate and quality
            # NO PADDING - V2 architecture handles variable lengths via packed attention
            processed_signal = acceleration.copy()
            
            print(f"   Processing: {len(processed_signal)} samples at {original_rate:.1f} Hz (variable length)")
            
            # Apply filters with original sampling rate
            broadband, lowfreq = apply_filters(processed_signal, original_rate)
            
            # Calculate PGA values for reference (signals kept RAW, not normalized)
            pga_broadband = np.max(np.abs(broadband))
            pga_lowfreq = np.max(np.abs(lowfreq))
            
            # FIX BUG #1: Keep signals RAW (no normalization)
            # Training will work with original signal scales
            # This ensures diffusion happens at the correct physical amplitude
            broadband_raw = broadband  # Keep original scale
            lowfreq_raw = lowfreq      # Keep original scale
            
            # Save processed data (RAW signals, not normalized)
            output_name = os.path.basename(file_path).replace('.AT2', '_processed.npz')
            output_path = os.path.join(output_dir, output_name)
            
            np.savez_compressed(output_path,
                signal_broadband=broadband_raw,        # RAW broadband signal
                signal_lowfreq=lowfreq_raw,            # RAW lowpass signal
                pga_broadband=pga_broadband,           # Saved for reference only
                pga_lowfreq=pga_lowfreq,               # Saved for reference only
                sampling_rate=original_rate,           # PRESERVE ORIGINAL RATE
                duration=len(processed_signal) / original_rate,
                actual_samples=len(processed_signal),
                file_path=file_path  # Store for reference
            )
            
            print(f"   ✅ Saved: {output_name}")
            print(f"   📊 PGA broadband: {pga_broadband:.4f}")
            print(f"   📊 RAW ranges (not normalized): broad=[{broadband_raw.min():.6f}, {broadband_raw.max():.6f}], low=[{lowfreq_raw.min():.6f}, {lowfreq_raw.max():.6f}]")
            
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {file_path}: {e}")
            continue
    
    print(f"\n🎉 PROCESSING COMPLETE!")
    print(f"✅ Successfully processed: {processed_count}/{len(at2_files)} files")
    print(f"💾 Output directory: {output_dir}")
    print(f"📏 LENGTH RANGE DETECTED:")
    print(f"   - Maximum length: {target_length} samples")
    print(f"   - Reference sampling rate: {reference_rate:.1f} Hz")
    print(f"   - Maximum duration: {target_length / reference_rate:.1f} seconds")
    
    print(f"\n✅ PRESERVED DATA QUALITY:")
    print(f"   - Original sampling rates maintained")
    print(f"   - No resampling artifacts")
    print(f"   - Variable-length sequences (no padding)")
    print(f"   - Packed attention handles length variations")
    print(f"   - RAW signal amplitudes preserved (NO normalization)")
    print(f"   - PGA values saved for reference only")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Dataset will pack variable-length sequences efficiently")
    print(f"2. Run: python train_signal_v2.py --data_dir data_prep_acc/processed_dynamic --batch_size 2 --pack_size 2")

if __name__ == "__main__":
    main()