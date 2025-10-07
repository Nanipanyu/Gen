#!/usr/bin/env python3
"""
Earthquake Signal Preprocessing for Diffusion Transformer Training

This module processes AT2 files from the PEER NGA database for broadband earthquake
ground motion signal generation. Features include:
- STA/LTA and Z-detector algorithms for event detection
- 60-second time windows with zero-padding for short records
- Data augmentation: random time shifts and horizontal component rotation
- Band-pass filtering (0.1-30 Hz) for broadband signals
- NPZ format output for PyTorch training pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import os
import glob
import random
from typing import List, Tuple, Dict, Optional

def read_at2(filename: str) -> Tuple[np.ndarray, float, int]:
    """
    Read AT2 file and extract acceleration data
    
    Args:
        filename: Path to AT2 file
        
    Returns:
        accel: Acceleration data in g
        dt: Time step in seconds
        npts: Number of data points
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Find line with NPTS and DT
    npts, dt = None, None
    for line in lines:
        if "NPTS" in line and "DT" in line:
            parts = line.replace("=", " ").replace(",", " ").split()
            npts = int(parts[parts.index("NPTS") + 1])
            dt = float(parts[parts.index("DT") + 1])
            break

    if npts is None or dt is None:
        raise ValueError("Could not find NPTS and DT in file header")

    # Extract numerical values (skip header lines)
    data = []
    for line in lines:
        # Split into numbers
        for val in line.strip().split():
            try:
                data.append(float(val))
            except ValueError:
                continue  # skip text parts

    # Only keep the last npts values
    accel = np.array(data[-npts:])
    
    if len(accel) < npts:
        print(f"Warning: Expected {npts} points, got {len(accel)}")
    
    return accel, dt, npts

def sta_lta_detector(signal: np.ndarray, dt: float, sta_len: float = 1.0, 
                    lta_len: float = 20.0, trigger_ratio: float = 3.0) -> Tuple[int, np.ndarray]:
    """
    STA/LTA (Short Term Average/Long Term Average) event detection
    
    Args:
        signal: Input acceleration signal
        dt: Time step in seconds
        sta_len: Short term average window length in seconds
        lta_len: Long term average window length in seconds
        trigger_ratio: STA/LTA ratio threshold for event detection
        
    Returns:
        trigger_idx: Index of event detection (or -1 if no event)
        sta_lta_ratio: STA/LTA ratio time series
    """
    sta_samples = int(sta_len / dt)
    lta_samples = int(lta_len / dt)
    
    if len(signal) < lta_samples:
        return -1, np.zeros_like(signal)
    
    # Compute squared signal for energy calculation
    signal_sq = signal ** 2
    
    # Initialize arrays
    sta_lta_ratio = np.zeros(len(signal))
    
    # Compute STA/LTA ratio
    for i in range(lta_samples, len(signal)):
        # Long term average (energy)
        lta = np.mean(signal_sq[i-lta_samples:i])
        
        # Short term average (energy)
        sta_start = max(0, i - sta_samples)
        sta = np.mean(signal_sq[sta_start:i])
        
        # Avoid division by zero
        if lta > 1e-12:
            sta_lta_ratio[i] = sta / lta
        else:
            sta_lta_ratio[i] = 0
    
    # Find first trigger point
    trigger_indices = np.where(sta_lta_ratio > trigger_ratio)[0]
    trigger_idx = trigger_indices[0] if len(trigger_indices) > 0 else -1
    
    return trigger_idx, sta_lta_ratio

def z_detector(signal: np.ndarray, dt: float, window_len: float = 5.0, 
               threshold: float = 3.0) -> Tuple[int, np.ndarray]:
    """
    Z-detector algorithm for event detection based on statistical analysis
    
    Args:
        signal: Input acceleration signal
        dt: Time step in seconds
        window_len: Analysis window length in seconds
        threshold: Z-score threshold for detection
        
    Returns:
        trigger_idx: Index of event detection (or -1 if no event)
        z_scores: Z-score time series
    """
    window_samples = int(window_len / dt)
    
    if len(signal) < window_samples:
        return -1, np.zeros_like(signal)
    
    # Compute absolute signal
    abs_signal = np.abs(signal)
    
    # Initialize z-scores
    z_scores = np.zeros(len(signal))
    
    # Compute rolling statistics and z-scores
    for i in range(window_samples, len(signal)):
        # Background window statistics
        bg_window = abs_signal[i-window_samples:i]
        bg_mean = np.mean(bg_window)
        bg_std = np.std(bg_window)
        
        # Current sample z-score
        if bg_std > 1e-12:
            z_scores[i] = (abs_signal[i] - bg_mean) / bg_std
        else:
            z_scores[i] = 0
    
    # Find first detection
    trigger_indices = np.where(z_scores > threshold)[0]
    trigger_idx = trigger_indices[0] if len(trigger_indices) > 0 else -1
    
    return trigger_idx, z_scores

def extract_60s_window(signal: np.ndarray, dt: float, trigger_idx: Optional[int] = None,
                      target_duration: float = 60.0) -> np.ndarray:
    """
    Extract 60-second window from signal, with zero-padding if needed
    
    Args:
        signal: Input signal
        dt: Time step in seconds
        trigger_idx: Event trigger index (if None, use signal start)
        target_duration: Target window duration in seconds
        
    Returns:
        windowed_signal: 60-second signal window
    """
    target_samples = int(target_duration / dt)
    
    if trigger_idx is None:
        # No trigger detected, start from beginning
        start_idx = 0
    else:
        # Start 5 seconds before trigger for context
        pre_event_samples = int(5.0 / dt)
        start_idx = max(0, trigger_idx - pre_event_samples)
    
    # Extract window
    end_idx = start_idx + target_samples
    
    if end_idx <= len(signal):
        # Signal is long enough
        windowed_signal = signal[start_idx:end_idx].copy()
    else:
        # Signal is too short, need padding
        available_samples = len(signal) - start_idx
        windowed_signal = np.zeros(target_samples)
        
        if available_samples > 0:
            windowed_signal[:available_samples] = signal[start_idx:]
        
        print(f"Signal padded: {available_samples}/{target_samples} samples available")
    
    return windowed_signal

def apply_random_time_shift(signal: np.ndarray, dt: float, max_shift_sec: float = 2.0) -> np.ndarray:
    """
    Apply random time shift for data augmentation
    
    Args:
        signal: Input signal
        dt: Time step in seconds
        max_shift_sec: Maximum shift in seconds
        
    Returns:
        shifted_signal: Time-shifted signal
    """
    max_shift_samples = int(max_shift_sec / dt)
    shift_samples = random.randint(-max_shift_samples, max_shift_samples)
    
    if shift_samples == 0:
        return signal.copy()
    
    shifted_signal = np.zeros_like(signal)
    
    if shift_samples > 0:
        # Shift right (delay)
        if shift_samples < len(signal):
            shifted_signal[shift_samples:] = signal[:-shift_samples]
    else:
        # Shift left (advance)
        abs_shift = abs(shift_samples)
        if abs_shift < len(signal):
            shifted_signal[:-abs_shift] = signal[abs_shift:]
    
    return shifted_signal

def rotate_horizontal_components(h1_signal: np.ndarray, h2_signal: np.ndarray, 
                               rotation_angle: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotate horizontal components for data augmentation
    
    Args:
        h1_signal: First horizontal component
        h2_signal: Second horizontal component  
        rotation_angle: Rotation angle in degrees (if None, random 0-360°)
        
    Returns:
        rotated_h1: Rotated first component
        rotated_h2: Rotated second component
    """
    if rotation_angle is None:
        rotation_angle = random.uniform(0, 360)
    
    # Convert to radians
    theta = np.radians(rotation_angle)
    
    # Rotation matrix
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    # Apply rotation
    rotated_h1 = cos_theta * h1_signal - sin_theta * h2_signal
    rotated_h2 = sin_theta * h1_signal + cos_theta * h2_signal
    
    return rotated_h1, rotated_h2

def butterworth_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """Apply Butterworth band-pass filter"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    if low >= 1.0 or high >= 1.0:
        raise ValueError(f"Filter frequencies must be less than Nyquist frequency ({nyquist} Hz)")
    
    b, a = signal.butter(order, [low, high], btype='band')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data, b, a

def butterworth_lowpass_filter(data, cutoff, fs, order=4):
    """Apply Butterworth low-pass filter"""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")
    
    b, a = signal.butter(order, normalized_cutoff, btype='low')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data, b, a

def butterworth_highpass_filter(data, cutoff, fs, order=4):
    """Apply Butterworth high-pass filter"""
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff / nyquist
    
    if normalized_cutoff >= 1.0:
        raise ValueError(f"Cutoff frequency must be less than Nyquist frequency ({nyquist} Hz)")
    
    b, a = signal.butter(order, normalized_cutoff, btype='high')
    filtered_data = signal.filtfilt(b, a, data)
    return filtered_data, b, a

def plot_four_stage_comparison(original, bandpass, lowpass, highpass, dt, filename):
    """Plot comparison showing all four filtering stages"""
    time = np.arange(len(original)) * dt
    
    # Larger vertical size and shared x-axis so only the bottom plot shows x-tick labels
    fig, axes = plt.subplots(4, 1, figsize=(20, 24), sharex=True)
    fig.suptitle(f'Multi-Stage Butterworth Filtering: {os.path.basename(filename)}', fontsize=16)
    
    # Convert to m/s² for plotting
    original_ms2 = original * 9.81
    bandpass_ms2 = bandpass * 9.81  
    lowpass_ms2 = lowpass * 9.81
    highpass_ms2 = highpass * 9.81
    
    # Common styling for axes
    y_label_kwargs = dict(labelpad=12, fontsize=11)
    title_kwargs = dict(fontsize=12)
    tick_y_kwargs = dict(labelsize=10)
    tick_x_kwargs = dict(labelsize=10)
    
    # Original signal
    axes[0].plot(time, original_ms2, 'b-', linewidth=0.8)
    axes[0].set_ylabel('Acceleration (m/s²)', **y_label_kwargs)
    axes[0].set_title(f'Original Signal (Max: {np.max(np.abs(original_ms2)):.6f} m/s²)', **title_kwargs)
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', which='both', labelbottom=False)
    axes[0].tick_params(**tick_y_kwargs)
    
    # Band-pass filtered signal
    axes[1].plot(time, bandpass_ms2, 'r-', linewidth=0.8)
    axes[1].set_ylabel('Acceleration (m/s²)', **y_label_kwargs)
    axes[1].set_title(f'After Band-pass Filter (0.1-30 Hz) (Max: {np.max(np.abs(bandpass_ms2)):.6f} m/s²)', **title_kwargs)
    axes[1].grid(True, alpha=0.3)
    axes[1].tick_params(axis='x', which='both', labelbottom=False)
    axes[1].tick_params(**tick_y_kwargs)
    
    # Low-pass filtered signal (final from two-stage)
    axes[2].plot(time, lowpass_ms2, 'g-', linewidth=0.8)
    axes[2].set_ylabel('Acceleration (m/s²)', **y_label_kwargs)
    axes[2].set_title(f'After Low-pass Filter (1 Hz) - Final Two-Stage (Max: {np.max(np.abs(lowpass_ms2)):.6f} m/s²)', **title_kwargs)
    axes[2].grid(True, alpha=0.3)
    axes[2].tick_params(axis='x', which='both', labelbottom=False)
    axes[2].tick_params(**tick_y_kwargs)
    
    # High-pass filtered signal (>1 Hz)
    axes[3].plot(time, highpass_ms2, 'm-', linewidth=0.8)
    axes[3].set_xlabel('Time (s)', fontsize=12)
    axes[3].set_ylabel('Acceleration (m/s²)', **y_label_kwargs)
    axes[3].set_title(f'High-pass Filter (>1 Hz) (Max: {np.max(np.abs(highpass_ms2)):.6f} m/s²)', **title_kwargs)
    axes[3].grid(True, alpha=0.3)
    axes[3].tick_params(**tick_x_kwargs)
    axes[3].tick_params(**tick_y_kwargs)
    
    # Adjust spacing to avoid label overlap and leave room for suptitle
    fig.subplots_adjust(top=0.92, hspace=0.5, bottom=0.06)
    plt.tight_layout(rect=[0, 0, 1, 0.92], pad=4.0)
    plt.show()

def plot_fas_comparison(original, bandpass, lowpass, highpass, fs, filename):
    """Plot Fourier Amplitude Spectrum comparison using periodogram"""
    
    # Calculate FAS using periodogram for each signal
    print(f"[FAS] Calculating Fourier Amplitude Spectrum...")
    
    freq_orig, fas_orig_pow = signal.periodogram(original, fs, scaling='spectrum')
    freq_band, fas_band_pow = signal.periodogram(bandpass, fs, scaling='spectrum') 
    freq_low, fas_low_pow = signal.periodogram(lowpass, fs, scaling='spectrum')
    freq_high, fas_high_pow = signal.periodogram(highpass, fs, scaling='spectrum')
    
    # Convert power spectrum to amplitude spectrum (take square root)
    fas_orig = np.sqrt(fas_orig_pow)
    fas_band = np.sqrt(fas_band_pow)
    fas_low = np.sqrt(fas_low_pow)
    fas_high = np.sqrt(fas_high_pow)
    
    # Create 2x2 subplot grid for individual FAS plots; make panels slightly smaller so labels don't overlap
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Fourier Amplitude Spectrum Analysis: {os.path.basename(filename)}', fontsize=14)
    
    # Helper to reduce x-tick label density on log axes
    def reduce_xticks(ax, max_ticks=6):
        ticks = ax.get_xticks()
        if len(ticks) > max_ticks:
            keep = np.linspace(0, len(ticks)-1, max_ticks).astype(int)
            ax.set_xticks(ticks[keep])
            ax.tick_params(axis='x', rotation=30)
        else:
            ax.tick_params(axis='x', rotation=30)

    # Plot helper to move legend outside
    def plot_log(ax, f, s, color, label):
        ax.loglog(f, s, color=color, linewidth=1.0, alpha=0.9)
        ax.grid(True, which='both', alpha=0.3)
        leg = ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0.)
        return ax
    
    # Original FAS
    axes[0,0] = plot_log(axes[0,0], freq_orig, fas_orig, 'b', 'Original')
    axes[0,0].axvline(0.1, color='red', linestyle='--', alpha=0.7)
    axes[0,0].axvline(1.0, color='green', linestyle='--', alpha=0.7)
    axes[0,0].axvline(30.0, color='red', linestyle='--', alpha=0.7)
    axes[0,0].set_title('Original Signal FAS', fontsize=11)
    axes[0,0].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[0,0].set_ylabel('FAS (g)', fontsize=10)
    reduce_xticks(axes[0,0])
    
    # Band-pass FAS
    axes[0,1] = plot_log(axes[0,1], freq_band, fas_band, 'r', 'Band-pass (0.1-30 Hz)')
    axes[0,1].axvline(0.1, color='red', linestyle='--', alpha=0.7)
    axes[0,1].axvline(30.0, color='red', linestyle='--', alpha=0.7)
    axes[0,1].set_title('Band-pass Filtered (0.1-30 Hz) FAS', fontsize=11)
    axes[0,1].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[0,1].set_ylabel('FAS (g)', fontsize=10)
    reduce_xticks(axes[0,1])
    
    # Low-pass FAS
    axes[1,0] = plot_log(axes[1,0], freq_low, fas_low, 'g', 'Low-pass (1 Hz)')
    axes[1,0].axvline(1.0, color='green', linestyle='--', alpha=0.7)
    axes[1,0].set_title('Low-pass Filtered (1 Hz) FAS - Final Two-Stage', fontsize=11)
    axes[1,0].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[1,0].set_ylabel('FAS (g)', fontsize=10)
    reduce_xticks(axes[1,0])
    
    # High-pass FAS
    axes[1,1] = plot_log(axes[1,1], freq_high, fas_high, 'm', 'High-pass (>1 Hz)')
    axes[1,1].axvline(1.0, color='green', linestyle='--', alpha=0.7)
    axes[1,1].set_title('High-pass Filtered (>1 Hz) FAS', fontsize=11)
    axes[1,1].set_xlabel('Frequency (Hz)', fontsize=10)
    axes[1,1].set_ylabel('FAS (g)', fontsize=10)
    reduce_xticks(axes[1,1])
    
    # Tight layout and move legends out to the right so they don't overlap
    plt.subplots_adjust(wspace=0.45, hspace=0.45, right=0.82, top=0.90)
    plt.show()
    
    # Print FAS analysis
    print(f"[FAS] Frequency Analysis:")
    print(f"   Peak frequency (Original): {freq_orig[np.argmax(fas_orig)]:.3f} Hz")
    print(f"   Peak frequency (Final): {freq_low[np.argmax(fas_low)]:.3f} Hz")

def plot_fas_overlay(original, bandpass, lowpass, highpass, fs, filename):
    """Plot all FAS overlaid for direct comparison"""
    
    freq_orig, fas_orig_pow = signal.periodogram(original, fs, scaling='spectrum')
    freq_band, fas_band_pow = signal.periodogram(bandpass, fs, scaling='spectrum')
    freq_low, fas_low_pow = signal.periodogram(lowpass, fs, scaling='spectrum')
    freq_high, fas_high_pow = signal.periodogram(highpass, fs, scaling='spectrum')
    
    # Convert to amplitude spectrum
    fas_orig = np.sqrt(fas_orig_pow)
    fas_band = np.sqrt(fas_band_pow)
    fas_low = np.sqrt(fas_low_pow)
    fas_high = np.sqrt(fas_high_pow)
    
    plt.figure(figsize=(18, 8))
    
    plt.loglog(freq_orig, fas_orig, 'b-', linewidth=1.2, alpha=0.9, label='Original Signal')
    plt.loglog(freq_band, fas_band, 'r-', linewidth=1.2, alpha=0.9, label='Band-pass (0.1-30 Hz)')
    plt.loglog(freq_low, fas_low, 'g-', linewidth=1.2, alpha=0.9, label='Low-pass (1 Hz) - Final')
    plt.loglog(freq_high, fas_high, 'm-', linewidth=1.2, alpha=0.9, label='High-pass (>1 Hz)')
    
    # Add vertical lines for filter cutoffs
    plt.axvline(0.1, color='red', linestyle='--', alpha=0.5)
    plt.axvline(30.0, color='red', linestyle='--', alpha=0.5)
    plt.axvline(1.0, color='green', linestyle=':', alpha=0.7, linewidth=2)
    
    plt.title(f'Fourier Amplitude Spectrum Overlay: {os.path.basename(filename)}', fontsize=14)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Fourier Amplitude Spectrum (g)', fontsize=12)
    # Place legend outside to the right
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)
    plt.grid(True, alpha=0.3)
    plt.xlim(0.01, fs/2)
    
    # Add frequency band annotations to left side so they don't collide with legend
    ylim = plt.ylim()
    plt.text(0.05, ylim[1]*0.6, 'Low freq\n(drift)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(2, ylim[1]*0.6, 'Structural\nResponse', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
    # Place high freq annotation lower and left so legend doesn't cover it
    plt.text(20, ylim[1]*0.05, 'High freq\n(noise)', ha='center', va='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1.0], pad=3.0)
    plt.show()

def save_all_filtered_data(original, bandpass, lowpass, highpass, dt, filename):
    """Save all filtered data to their respective folders"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    time_array = np.arange(len(original)) * dt
    
    # Create folders if they don't exist
    folders = ['bandpass', 'lowpass', 'highpass']
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
    
    saved_files = []
    
    # Save band-pass filtered data
    bandpass_file = os.path.join('bandpass', f"{base_name}_bandpass_filtered.txt")
    bandpass_ms2 = bandpass * 9.81
    with open(bandpass_file, 'w') as f:
        f.write(f"# Band-pass filtered acceleration data (0.1-30 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth Band-pass 0.1-30 Hz (Order 4)\n")
        f.write(f"# Time step: {dt} seconds\n")
        f.write(f"# Units: m/s²\n")
        f.write(f"# Time(s)\tAcceleration(m/s²)\n")
        for t, a in zip(time_array, bandpass_ms2):
            f.write(f"{t:.6f}\t{a:.8f}\n")
    saved_files.append(bandpass_file)
    
    # Save low-pass filtered data
    lowpass_file = os.path.join('lowpass', f"{base_name}_lowpass_filtered.txt")
    lowpass_ms2 = lowpass * 9.81
    with open(lowpass_file, 'w') as f:
        f.write(f"# Low-pass filtered acceleration data (1 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth Low-pass 1 Hz (Order 4)\n")
        f.write(f"# Applied after band-pass filtering (0.1-30 Hz)\n")
        f.write(f"# Time step: {dt} seconds\n")
        f.write(f"# Units: m/s²\n")
        f.write(f"# Time(s)\tAcceleration(m/s²)\n")
        for t, a in zip(time_array, lowpass_ms2):
            f.write(f"{t:.6f}\t{a:.8f}\n")
    saved_files.append(lowpass_file)
    
    # Save high-pass filtered data
    highpass_file = os.path.join('highpass', f"{base_name}_highpass_filtered.txt")
    highpass_ms2 = highpass * 9.81
    with open(highpass_file, 'w') as f:
        f.write(f"# High-pass filtered acceleration data (>1 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth High-pass 1 Hz (Order 4)\n")
        f.write(f"# Time step: {dt} seconds\n")
        f.write(f"# Units: m/s²\n")
        f.write(f"# Time(s)\tAcceleration(m/s²)\n")
        for t, a in zip(time_array, highpass_ms2):
            f.write(f"{t:.6f}\t{a:.8f}\n")
    saved_files.append(highpass_file)
    
    return saved_files

def save_fas_data(original, bandpass, lowpass, highpass, fs, filename):
    """Save FAS data to FAS folder"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    # Create FAS folder if it doesn't exist
    os.makedirs('FAS', exist_ok=True)
    
    # Calculate FAS using periodogram
    freq_orig, fas_orig_pow = signal.periodogram(original, fs, scaling='spectrum')
    freq_band, fas_band_pow = signal.periodogram(bandpass, fs, scaling='spectrum')
    freq_low, fas_low_pow = signal.periodogram(lowpass, fs, scaling='spectrum')
    freq_high, fas_high_pow = signal.periodogram(highpass, fs, scaling='spectrum')
    
    # Convert to amplitude spectrum (take square root)
    fas_orig = np.sqrt(fas_orig_pow)
    fas_band = np.sqrt(fas_band_pow)
    fas_low = np.sqrt(fas_low_pow)
    fas_high = np.sqrt(fas_high_pow)
    
    saved_files = []
    
    # Save original FAS
    orig_fas_file = os.path.join('FAS', f"{base_name}_original_fas.txt")
    with open(orig_fas_file, 'w') as f:
        f.write(f"# Original signal Fourier Amplitude Spectrum\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Method: Periodogram with spectrum scaling\n")
        f.write(f"# Sampling frequency: {fs} Hz\n")
        f.write(f"# Units: g\n")
        f.write(f"# Frequency(Hz)\tFAS(g)\n")
        for freq, fas in zip(freq_orig, fas_orig):
            f.write(f"{freq:.6f}\t{fas:.8e}\n")
    saved_files.append(orig_fas_file)
    
    # Save band-pass FAS
    band_fas_file = os.path.join('FAS', f"{base_name}_bandpass_fas.txt")
    with open(band_fas_file, 'w') as f:
        f.write(f"# Band-pass filtered signal Fourier Amplitude Spectrum (0.1-30 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth Band-pass 0.1-30 Hz (Order 4)\n")
        f.write(f"# Method: Periodogram with spectrum scaling\n")
        f.write(f"# Sampling frequency: {fs} Hz\n")
        f.write(f"# Units: g\n")
        f.write(f"# Frequency(Hz)\tFAS(g)\n")
        for freq, fas in zip(freq_band, fas_band):
            f.write(f"{freq:.6f}\t{fas:.8e}\n")
    saved_files.append(band_fas_file)
    
    # Save low-pass FAS
    low_fas_file = os.path.join('FAS', f"{base_name}_lowpass_fas.txt")
    with open(low_fas_file, 'w') as f:
        f.write(f"# Low-pass filtered signal Fourier Amplitude Spectrum (1 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth Low-pass 1 Hz (Order 4) after band-pass\n")
        f.write(f"# Method: Periodogram with spectrum scaling\n")
        f.write(f"# Sampling frequency: {fs} Hz\n")
        f.write(f"# Units: g\n")
        f.write(f"# Frequency(Hz)\tFAS(g)\n")
        for freq, fas in zip(freq_low, fas_low):
            f.write(f"{freq:.6f}\t{fas:.8e}\n")
    saved_files.append(low_fas_file)
    
    # Save high-pass FAS
    high_fas_file = os.path.join('FAS', f"{base_name}_highpass_fas.txt")
    with open(high_fas_file, 'w') as f:
        f.write(f"# High-pass filtered signal Fourier Amplitude Spectrum (>1 Hz)\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Filter: Butterworth High-pass 1 Hz (Order 4)\n")
        f.write(f"# Method: Periodogram with spectrum scaling\n")
        f.write(f"# Sampling frequency: {fs} Hz\n")
        f.write(f"# Units: g\n")
        f.write(f"# Frequency(Hz)\tFAS(g)\n")
        for freq, fas in zip(freq_high, fas_high):
            f.write(f"{freq:.6f}\t{fas:.8e}\n")
    saved_files.append(high_fas_file)
    
    return saved_files

def save_filtered_data(final_accel, dt, filename):
    """Save the final filtered data"""
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_file = f"{base_name}_filtered_two_stage.txt"
    
    final_ms2 = final_accel * 9.81
    time_array = np.arange(len(final_ms2)) * dt
    
    with open(output_file, 'w') as f:
        f.write(f"# Two-stage filtered acceleration data\n")
        f.write(f"# Original file: {os.path.basename(filename)}\n")
        f.write(f"# Stage 1: Band-pass filter 0.1-30 Hz (Order 4)\n")
        f.write(f"# Stage 2: Low-pass filter 1.0 Hz (Order 4)\n")
        f.write(f"# Time step: {dt} seconds\n")
        f.write(f"# Units: m/s²\n")
        f.write(f"# Time(s)\tAcceleration(m/s²)\n")
        
        for t, a in zip(time_array, final_ms2):
            f.write(f"{t:.6f}\t{a:.8f}\n")
    
    return output_file

def process_earthquake_signal(accel: np.ndarray, dt: float, apply_augmentation: bool = True) -> Dict:
    """
    Process earthquake signal with event detection, windowing, and augmentation
    ONLY does: STA/LTA detection, 60s windowing, time shifts, band-pass filtering (0.1-30 Hz)
    
    Args:
        accel: Raw acceleration signal in g
        dt: Time step in seconds
        apply_augmentation: Whether to apply data augmentation
        
    Returns:
        processed_data: Dictionary with processed broadband signal and metadata
    """
    fs = 1.0 / dt
    
    # Step 1: Event detection using STA/LTA
    trigger_idx, sta_lta_ratio = sta_lta_detector(accel, dt)
    
    # Also try Z-detector as backup
    z_trigger_idx, z_scores = z_detector(accel, dt)
    
    # Use STA/LTA trigger if available, otherwise Z-detector
    final_trigger = trigger_idx if trigger_idx != -1 else z_trigger_idx
    
    print(f"   Event detection - STA/LTA trigger: {trigger_idx}, Z-detector: {z_trigger_idx}")
    
    # Step 2: Extract 60-second window
    windowed_signal = extract_60s_window(accel, dt, final_trigger)
    
    # Step 3: Apply band-pass filtering (0.1-30 Hz) ONLY
    # Note: Low-pass filtering and PGA calculation will be handled in signal_datasets.py
    broadband_signal, _, _ = butterworth_bandpass_filter(windowed_signal, 0.1, 30.0, fs, 4)
    
    return {
        'original_windowed': windowed_signal,    # Raw 60s windowed signal
        'broadband': broadband_signal,           # Band-pass filtered (0.1-30 Hz)
        'trigger_idx': final_trigger,
        'sta_lta_ratio': sta_lta_ratio,
        'z_scores': z_scores,
        'sample_rate': fs,
        'dt': dt,
        'duration': len(windowed_signal) * dt,
        'augmentation_applied': []
    }

def process_at2_for_training(filename: str, output_dir: str, apply_augmentation: bool = True, 
                           num_augmentations: int = 3) -> List[str]:
    """
    Process AT2 file for diffusion transformer training
    
    Args:
        filename: Path to AT2 file
        output_dir: Output directory for NPZ files
        apply_augmentation: Whether to apply data augmentation
        num_augmentations: Number of augmented versions to create
        
    Returns:
        saved_files: List of saved NPZ file paths
    """
    print(f"\n{'='*70}")
    print(f"Processing for training: {os.path.basename(filename)}")
    print(f"{'='*70}")
    
    saved_files = []
    
    try:
        # Read AT2 file
        accel, dt, npts = read_at2(filename)
        fs = 1.0 / dt
        
        print(f"[INFO] File loaded successfully")
        print(f"   Duration: {npts*dt:.2f} seconds")
        print(f"   Sampling frequency: {fs:.1f} Hz")
        print(f"   Original max: {np.max(np.abs(accel))*9.81:.6f} m/s²")
        
        # Create base filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        
        # Process original signal
        processed = process_earthquake_signal(accel, dt, apply_augmentation=False)
        
        # Save original processed signal (only broadband, no low-pass or PGA)
        original_file = os.path.join(output_dir, f"{base_name}_original.npz")
        np.savez(original_file,
                signal_raw_windowed=processed['original_windowed'],  # Raw 60s windowed
                signal_broadband=processed['broadband'],             # Band-pass (0.1-30 Hz)
                sample_rate=processed['sample_rate'],
                dt=processed['dt'],
                duration=processed['duration'],
                trigger_idx=processed['trigger_idx'],
                original_file=filename,
                augmentation_type='original',
                augmentation_applied=processed['augmentation_applied'])
        saved_files.append(original_file)
        
        print(f"   Saved original: {os.path.basename(original_file)}")
        print(f"   Duration: {processed['duration']:.1f}s, Trigger: {processed['trigger_idx']}")
        
        return saved_files
        
    except Exception as e:
        print(f"[ERROR] Failed to process {filename}: {e}")
        return []

def process_horizontal_pair(h1_file: str, h2_file: str, output_dir: str, 
                          apply_rotation: bool = True, num_rotations: int = 5,
                          apply_time_shifts: bool = True, num_time_shifts: int = 3) -> List[str]:
    """
    Process horizontal component pair with rotation and time shift augmentation
    
    This function implements random rotation of horizontal components (E-W, N-S)
    which is a key augmentation for earthquake data.
    
    Args:
        h1_file: First horizontal component AT2 file (E-W)
        h2_file: Second horizontal component AT2 file (N-S)
        output_dir: Output directory
        apply_rotation: Whether to apply rotation augmentation
        num_rotations: Number of rotated versions to create
        apply_time_shifts: Whether to apply time shift augmentation
        num_time_shifts: Number of time shifted versions
        
    Returns:
        saved_files: List of saved NPZ file paths
    """
    print(f"\n{'='*70}")
    print(f"Processing horizontal pair with rotation:")
    print(f"  H1/E-W: {os.path.basename(h1_file)}")
    print(f"  H2/N-S: {os.path.basename(h2_file)}")
    print(f"{'='*70}")
    
    saved_files = []
    
    try:
        # Read both horizontal components
        h1_accel, dt1, npts1 = read_at2(h1_file)
        h2_accel, dt2, npts2 = read_at2(h2_file)
        
        # Ensure same sampling rate and length
        if abs(dt1 - dt2) > 1e-6:
            print(f"Warning: Different sampling rates - H1: {1/dt1:.1f} Hz, H2: {1/dt2:.1f} Hz")
            return []
        
        # Use minimum length
        min_length = min(len(h1_accel), len(h2_accel))
        h1_accel = h1_accel[:min_length]
        h2_accel = h2_accel[:min_length]
        dt = dt1
        fs = 1.0 / dt
        
        # Create base filename from h1 file
        base_name = os.path.splitext(os.path.basename(h1_file))[0]
        for suffix in ['_H1', '_E', '_EW', '_01']:
            base_name = base_name.replace(suffix, '')
        
        # Event detection on first component to align both
        trigger_idx, _, _ = sta_lta_detector(h1_accel, dt)
        z_trigger_idx, _ = z_detector(h1_accel, dt)
        final_trigger = trigger_idx if trigger_idx != -1 else z_trigger_idx
        
        print(f"   Event detection - Trigger at index: {final_trigger}")
        
        # Extract 60s windows from both components (synchronized)
        h1_windowed = extract_60s_window(h1_accel, dt, final_trigger)
        h2_windowed = extract_60s_window(h2_accel, dt, final_trigger)
        
        # Save original components (0° rotation)
        for comp_name, windowed in [('H1', h1_windowed), ('H2', h2_windowed)]:
            # Apply band-pass filtering
            broadband, _, _ = butterworth_bandpass_filter(windowed, 0.1, 30.0, fs, 4)
            
            comp_file = os.path.join(output_dir, f"{base_name}_{comp_name}_original.npz")
            np.savez(comp_file,
                    signal_raw_windowed=windowed,
                    signal_broadband=broadband,
                    sample_rate=fs,
                    dt=dt,
                    duration=len(windowed) * dt,
                    trigger_idx=final_trigger,
                    original_file=h1_file if comp_name == 'H1' else h2_file,
                    component=comp_name,
                    rotation_angle=0.0,
                    augmentation_type='original_horizontal',
                    augmentation_applied=[])
            saved_files.append(comp_file)
        
        print(f"   Saved original H1 and H2 components")
        
        print(f"   Generated {len(saved_files)} files from horizontal pair")
        return saved_files
        
    except Exception as e:
        print(f"[ERROR] Failed to process horizontal pair: {e}")
        return []

def process_all_for_training(input_dir: str = ".", output_dir: str = "processed_signals", 
                           apply_augmentation: bool = False, num_augmentations: int = 3):
    """
    Process all AT2 files for diffusion transformer training
    
    Args:
        input_dir: Directory containing AT2 files
        output_dir: Output directory for processed NPZ files
        apply_augmentation: Whether to apply data augmentation
        num_augmentations: Number of augmented versions per file
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all AT2 files
    at2_files = glob.glob(os.path.join(input_dir, "*.AT2"))
    
    if not at2_files:
        print(f"No AT2 files found in {input_dir}")
        return
    
    at2_files.sort()
    
    print("🚀 Earthquake Signal Processing for Diffusion Transformer Training")
    print("Features: STA/LTA detection, 60s windowing, band-pass filtering")
    print(f"Found {len(at2_files)} AT2 files")
    print(f"Output directory: {output_dir}")
    
    # Group files by station for horizontal pair processing
    station_groups = {}
    individual_files = []
    
    for filename in at2_files:
        base_name = os.path.basename(filename)
        
        # Try to identify horizontal pairs (common naming patterns)
        if '_H1' in base_name or '_E' in base_name:
            station_id = base_name.replace('_H1', '').replace('_E', '').replace('.AT2', '')
            if station_id not in station_groups:
                station_groups[station_id] = {}
            station_groups[station_id]['h1'] = filename
        elif '_H2' in base_name or '_N' in base_name:
            station_id = base_name.replace('_H2', '').replace('_N', '').replace('.AT2', '')
            if station_id not in station_groups:
                station_groups[station_id] = {}
            station_groups[station_id]['h2'] = filename
        else:
            individual_files.append(filename)
    
    # Process horizontal pairs with rotation
    total_saved = 0
    pair_count = 0
    
    for station_id, components in station_groups.items():
        if 'h1' in components and 'h2' in components:
            print(f"\n[PAIR] Processing horizontal pair: {station_id}")
            saved_files = process_horizontal_pair(
                components['h1'], components['h2'], output_dir,
                apply_rotation=False, num_rotations=0
            )
            total_saved += len(saved_files)
            pair_count += 1
        else:
            # Add incomplete pairs to individual processing
            if 'h1' in components:
                individual_files.append(components['h1'])
            if 'h2' in components:
                individual_files.append(components['h2'])
    
    # Process individual files
    individual_count = 0
    for filename in individual_files:
        print(f"\n[INDIVIDUAL] Processing: {os.path.basename(filename)}")
        saved_files = process_at2_for_training(
            filename, output_dir, 
            apply_augmentation=False,
            num_augmentations=0
        )
        total_saved += len(saved_files)
        if saved_files:
            individual_count += 1
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Total AT2 files found: {len(at2_files)}")
    print(f"Horizontal pairs processed: {pair_count}")
    print(f"Individual files processed: {individual_count}")
    print(f"Total NPZ files generated: {total_saved}")
    print(f"Output directory: {output_dir}")
    
    if total_saved > 0:
        print(f"\n✅ Signal processing completed!")
        print(f"Ready for diffusion transformer training!")
        print(f"\nNext steps:")
        print(f"1. Update signal_datasets.py to load from: {output_dir}")
        print(f"2. Run training with: python train_signal.py")
    else:
        print("❌ No files were processed successfully")

def process_all_at2_files():
    """Legacy function for compatibility - redirects to new training-focused processing"""
    print("Note: Using new training-focused processing pipeline")
    process_all_for_training()

if __name__ == "__main__":
    process_all_at2_files()
