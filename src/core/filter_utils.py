"""
Signal filtering utilities.

This module implements the exact filtering algorithm from the validated
legacy MATLAB code for sigma gradient calculations.
"""

import numpy as np
from typing import List, Union, Optional
import logging

from src.core.constants import FILTER_SAMPLING_FREQUENCY, FILTER_CUTOFF_FREQUENCY

logger = logging.getLogger(__name__)


def apply_filter(
        input_data: Union[List[float], np.ndarray],
        fs: int = FILTER_SAMPLING_FREQUENCY,
        fc: int = FILTER_CUTOFF_FREQUENCY
) -> List[float]:
    """
    Apply first-order digital filter matching MATLAB's my_filtfiltfd2.

    This implementation exactly matches the legacy validated calculations.
    It applies the filter in forward and backward directions for zero
    phase distortion.

    Args:
        input_data: Input signal to filter
        fs: Sampling frequency (default: 100 Hz)
        fc: Cutoff frequency (default: 80 Hz)

    Returns:
        Filtered signal as list

    Note:
        This is the exact implementation from the validated MATLAB code:
        - Forward pass: output(i) = output(i-1) + fc/fs * (input(i) - output(i-1))
        - Backward pass: output2(i) = output2(i+1) + fc/fs * (output(i) - output2(i+1))
    """
    # Convert to numpy array
    input_array = np.array(input_data, dtype=float)

    if len(input_array) == 0:
        return []

    # Calculate filter coefficient
    alpha = fc / fs

    # Forward pass - exactly like MATLAB
    output = np.zeros_like(input_array, dtype=float)
    output[0] = input_array[0]

    for i in range(1, len(input_array)):
        output[i] = output[i - 1] + alpha * (input_array[i] - output[i - 1])

    # Backward pass - exactly like MATLAB
    output2 = np.zeros_like(input_array, dtype=float)
    output2[-1] = output[-1]

    for i in range(len(input_array) - 2, -1, -1):
        output2[i] = output2[i + 1] + alpha * (output[i] - output2[i + 1])

    return output2.tolist()


def validate_filter_parameters(fs: int, fc: int) -> bool:
    """
    Validate filter parameters.

    Args:
        fs: Sampling frequency
        fc: Cutoff frequency

    Returns:
        True if parameters are valid
    """
    if fs <= 0:
        logger.error(f"Invalid sampling frequency: {fs}")
        return False

    if fc <= 0:
        logger.error(f"Invalid cutoff frequency: {fc}")
        return False

    if fc >= fs / 2:
        logger.error(f"Cutoff frequency {fc} must be less than Nyquist frequency {fs / 2}")
        return False

    return True


def calculate_filter_response(
        fs: int = FILTER_SAMPLING_FREQUENCY,
        fc: int = FILTER_CUTOFF_FREQUENCY,
        num_points: int = 1000
) -> tuple:
    """
    Calculate filter frequency response for visualization.

    Args:
        fs: Sampling frequency
        fc: Cutoff frequency
        num_points: Number of frequency points

    Returns:
        Tuple of (frequencies, magnitude_response)
    """
    # Calculate filter coefficient
    alpha = fc / fs

    # Frequency points from 0 to Nyquist
    frequencies = np.linspace(0, fs / 2, num_points)

    # Calculate transfer function magnitude
    # H(z) = alpha / (1 - (1-alpha)z^-1)
    # Applied twice (forward and backward)

    magnitude = []
    for f in frequencies:
        if f == 0:
            mag = 1.0
        else:
            # z = e^(j*2*pi*f/fs)
            z = np.exp(1j * 2 * np.pi * f / fs)
            h_single = alpha / (1 - (1 - alpha) / z)
            # Applied twice
            mag = abs(h_single) ** 2
        magnitude.append(mag)

    return frequencies, np.array(magnitude)


def smooth_data(
        data: Union[List[float], np.ndarray],
        window_size: int = 3
) -> List[float]:
    """
    Apply simple moving average smoothing.

    This is an alternative to the digital filter for comparison.

    Args:
        data: Input data
        window_size: Size of moving average window

    Returns:
        Smoothed data
    """
    if window_size < 1:
        raise ValueError("Window size must be at least 1")

    data_array = np.array(data, dtype=float)

    if len(data_array) == 0:
        return []

    if window_size == 1:
        return data_array.tolist()

    # Pad the data for edge handling
    pad_width = window_size // 2
    padded = np.pad(data_array, pad_width, mode='edge')

    # Apply moving average
    smoothed = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')

    return smoothed.tolist()


def remove_outliers(
        data: Union[List[float], np.ndarray],
        n_sigma: float = 3.0
) -> tuple:
    """
    Remove outliers using z-score method.

    Args:
        data: Input data
        n_sigma: Number of standard deviations for outlier threshold

    Returns:
        Tuple of (cleaned_data, outlier_indices)
    """
    data_array = np.array(data, dtype=float)

    if len(data_array) < 3:
        return data_array.tolist(), []

    # Calculate z-scores
    mean = np.mean(data_array)
    std = np.std(data_array)

    if std == 0:
        return data_array.tolist(), []

    z_scores = np.abs((data_array - mean) / std)

    # Find outliers
    outlier_mask = z_scores > n_sigma
    outlier_indices = np.where(outlier_mask)[0].tolist()

    # Remove outliers
    cleaned_data = data_array[~outlier_mask].tolist()

    return cleaned_data, outlier_indices


def interpolate_missing_data(
        positions: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        missing_indices: Optional[List[int]] = None
) -> List[float]:
    """
    Interpolate missing or removed data points.

    Args:
        positions: Position values
        values: Data values (may contain NaN)
        missing_indices: Indices of missing data

    Returns:
        Interpolated data
    """
    pos_array = np.array(positions, dtype=float)
    val_array = np.array(values, dtype=float)

    if missing_indices is None:
        # Find NaN values
        missing_mask = np.isnan(val_array)
        missing_indices = np.where(missing_mask)[0]
    else:
        missing_mask = np.zeros(len(val_array), dtype=bool)
        missing_mask[missing_indices] = True

    if len(missing_indices) == 0:
        return val_array.tolist()

    # Get valid data points
    valid_mask = ~missing_mask
    valid_positions = pos_array[valid_mask]
    valid_values = val_array[valid_mask]

    if len(valid_values) < 2:
        # Not enough points for interpolation
        logger.warning("Not enough valid points for interpolation")
        return val_array.tolist()

    # Interpolate missing values
    interpolated = np.interp(pos_array, valid_positions, valid_values)

    return interpolated.tolist()