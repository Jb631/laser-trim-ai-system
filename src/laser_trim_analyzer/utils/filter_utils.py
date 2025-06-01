"""
Signal filtering utilities for laser trim data processing.

Implements various filtering techniques to clean and smooth measurement data.
"""

import numpy as np
from scipy import signal, interpolate
from typing import List, Optional, Tuple, Type, Unionimport logging

logger = logging.getLogger(__name__)


def apply_filter(
        data: Union[List[float], np.ndarray],
        filter_type: str = "butterworth",
        cutoff_freq: float = 80.0,
        sampling_freq: float = 100.0,
        order: int = 4
) -> np.ndarray:
    """
    Apply digital filter to measurement data.

    This implements the same filtering used in the MATLAB version for consistency.

    Args:
        data: Input data to filter
        filter_type: Type of filter ("butterworth", "chebyshev", "bessel")
        cutoff_freq: Cutoff frequency in Hz
        sampling_freq: Sampling frequency in Hz
        order: Filter order

    Returns:
        Filtered data array
    """
    if not data or len(data) < order + 1:
        logger.warning(f"Insufficient data points ({len(data)}) for order {order} filter")
        return np.array(data)

    # Convert to numpy array
    data = np.asarray(data, dtype=float)

    # Handle NaN values
    nan_mask = np.isnan(data)
    if np.any(nan_mask):
        logger.debug(f"Found {np.sum(nan_mask)} NaN values, interpolating")
        data = interpolate_missing_data(data)

    # Normalize frequencies
    nyquist_freq = sampling_freq / 2
    normalized_cutoff = cutoff_freq / nyquist_freq

    # Ensure normalized frequency is valid
    if normalized_cutoff >= 1:
        logger.warning(f"Cutoff frequency {cutoff_freq} too high for sampling rate {sampling_freq}")
        normalized_cutoff = 0.99

    try:
        # Design filter based on type
        if filter_type.lower() == "butterworth":
            b, a = signal.butter(order, normalized_cutoff, btype='low', analog=False)
        elif filter_type.lower() == "chebyshev":
            b, a = signal.cheby1(order, 0.5, normalized_cutoff, btype='low', analog=False)
        elif filter_type.lower() == "bessel":
            b, a = signal.bessel(order, normalized_cutoff, btype='low', analog=False)
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        # Apply filter using filtfilt for zero-phase filtering
        filtered_data = signal.filtfilt(b, a, data)

        return filtered_data

    except Exception as e:
        logger.error(f"Filter application failed: {e}")
        return data


def smooth_data(
        data: Union[List[float], np.ndarray],
        window_size: int = 5,
        method: str = "savgol",
        polyorder: int = 2
) -> np.ndarray:
    """
    Smooth data using various methods.

    Args:
        data: Input data to smooth
        window_size: Size of the smoothing window
        method: Smoothing method ("savgol", "moving_average", "gaussian", "median")
        polyorder: Polynomial order for Savitzky-Golay filter

    Returns:
        Smoothed data array
    """
    data = np.asarray(data, dtype=float)

    if len(data) < window_size:
        logger.warning(f"Data length ({len(data)}) less than window size ({window_size})")
        return data

    # Ensure odd window size
    if window_size % 2 == 0:
        window_size += 1

    try:
        if method == "savgol":
            # Savitzky-Golay filter
            smoothed = signal.savgol_filter(data, window_size, polyorder)

        elif method == "moving_average":
            # Simple moving average
            kernel = np.ones(window_size) / window_size
            # Pad data to handle edges
            padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')

        elif method == "gaussian":
            # Gaussian weighted average
            sigma = window_size / 4  # Standard deviation
            kernel = signal.windows.gaussian(window_size, sigma)
            kernel = kernel / kernel.sum()
            padded = np.pad(data, (window_size // 2, window_size // 2), mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')

        elif method == "median":
            # Median filter
            smoothed = signal.medfilt(data, kernel_size=window_size)

        else:
            raise ValueError(f"Unknown smoothing method: {method}")

        return smoothed

    except Exception as e:
        logger.error(f"Smoothing failed: {e}")
        return data


def remove_outliers(
        data: Union[List[float], np.ndarray],
        method: str = "iqr",
        threshold: float = 1.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from data.

    Args:
        data: Input data
        method: Outlier detection method ("iqr", "zscore", "isolation")
        threshold: Threshold for outlier detection

    Returns:
        Tuple of (cleaned_data, outlier_mask)
    """
    data = np.asarray(data, dtype=float)
    outlier_mask = np.zeros(len(data), dtype=bool)

    if method == "iqr":
        # Interquartile range method
        q1 = np.nanpercentile(data, 25)
        q3 = np.nanpercentile(data, 75)
        iqr = q3 - q1

        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr

        outlier_mask = (data < lower_bound) | (data > upper_bound)

    elif method == "zscore":
        # Z-score method
        mean = np.nanmean(data)
        std = np.nanstd(data)

        if std > 0:
            z_scores = np.abs((data - mean) / std)
            outlier_mask = z_scores > threshold

    elif method == "isolation":
        # Modified z-score using median absolute deviation
        median = np.nanmedian(data)
        mad = np.nanmedian(np.abs(data - median))

        if mad > 0:
            modified_z_scores = 0.6745 * (data - median) / mad
            outlier_mask = np.abs(modified_z_scores) > threshold

    # Replace outliers with interpolated values
    cleaned_data = data.copy()
    if np.any(outlier_mask):
        cleaned_data[outlier_mask] = np.nan
        cleaned_data = interpolate_missing_data(cleaned_data)
        logger.debug(f"Removed {np.sum(outlier_mask)} outliers using {method} method")

    return cleaned_data, outlier_mask


def interpolate_missing_data(
        data: Union[List[float], np.ndarray],
        method: str = "linear"
) -> np.ndarray:
    """
    Interpolate missing (NaN) values in data.

    Args:
        data: Input data with potential NaN values
        method: Interpolation method ("linear", "cubic", "nearest")

    Returns:
        Data with interpolated values
    """
    data = np.asarray(data, dtype=float)

    if not np.any(np.isnan(data)):
        return data

    # Get indices of valid and invalid data
    valid_mask = ~np.isnan(data)

    if not np.any(valid_mask):
        # All data is NaN
        logger.warning("All data values are NaN")
        return np.zeros_like(data)

    if np.sum(valid_mask) < 2:
        # Not enough valid points for interpolation
        logger.warning("Insufficient valid data points for interpolation")
        return np.nan_to_num(data, nan=np.nanmean(data))

    # Create interpolation function
    indices = np.arange(len(data))
    valid_indices = indices[valid_mask]
    valid_data = data[valid_mask]

    try:
        if method == "linear":
            interp_func = interpolate.interp1d(
                valid_indices, valid_data,
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
        elif method == "cubic":
            if len(valid_data) >= 4:
                interp_func = interpolate.interp1d(
                    valid_indices, valid_data,
                    kind='cubic',
                    fill_value='extrapolate',
                    bounds_error=False
                )
            else:
                # Fall back to linear if not enough points
                interp_func = interpolate.interp1d(
                    valid_indices, valid_data,
                    kind='linear',
                    fill_value='extrapolate',
                    bounds_error=False
                )
        elif method == "nearest":
            interp_func = interpolate.interp1d(
                valid_indices, valid_data,
                kind='nearest',
                fill_value='extrapolate',
                bounds_error=False
            )
        else:
            raise ValueError(f"Unknown interpolation method: {method}")

        # Apply interpolation
        interpolated_data = interp_func(indices)

        return interpolated_data

    except Exception as e:
        logger.error(f"Interpolation failed: {e}")
        # Fall back to forward fill
        return pd.Series(data).fillna(method='ffill').fillna(method='bfill').values


def apply_end_point_filter(
        data: Union[List[float], np.ndarray],
        points_to_remove: int = 7
) -> np.ndarray:
    """
    Remove specified number of points from each end of the data.

    This is used in MATLAB processing to remove edge effects.

    Args:
        data: Input data
        points_to_remove: Number of points to remove from each end

    Returns:
        Filtered data with endpoints removed
    """
    data = np.asarray(data)

    if len(data) <= 2 * points_to_remove:
        logger.warning(
            f"Data length ({len(data)}) too short for removing "
            f"{points_to_remove} points from each end"
        )
        return data

    return data[points_to_remove:-points_to_remove]


def calculate_gradient(
        positions: Union[List[float], np.ndarray],
        values: Union[List[float], np.ndarray],
        step_size: int = 3
) -> np.ndarray:
    """
    Calculate gradient using specified step size.

    This implements the MATLAB-compatible gradient calculation.

    Args:
        positions: Position values
        values: Corresponding measurement values
        step_size: Step size for gradient calculation

    Returns:
        Array of gradient values
    """
    positions = np.asarray(positions)
    values = np.asarray(values)

    if len(positions) != len(values):
        raise ValueError("Positions and values must have same length")

    if len(positions) < step_size + 1:
        logger.warning(f"Insufficient data points for step size {step_size}")
        return np.array([])

    gradients = []

    for i in range(len(positions) - step_size):
        dx = positions[i + step_size] - positions[i]
        dy = values[i + step_size] - values[i]

        if dx != 0:
            gradient = dy / dx
            gradients.append(gradient)
        else:
            logger.warning(f"Zero position difference at index {i}")
            gradients.append(0.0)

    return np.array(gradients)


# QA-specific filter presets
class FilterPresets:
    """Predefined filter configurations for potentiometer QA."""

    STANDARD = {
        "filter_type": "butterworth",
        "cutoff_freq": 80.0,
        "sampling_freq": 100.0,
        "order": 4
    }

    HIGH_PRECISION = {
        "filter_type": "butterworth",
        "cutoff_freq": 60.0,
        "sampling_freq": 100.0,
        "order": 6
    }

    NOISE_REDUCTION = {
        "filter_type": "chebyshev",
        "cutoff_freq": 50.0,
        "sampling_freq": 100.0,
        "order": 5
    }

    MINIMAL = {
        "filter_type": "butterworth",
        "cutoff_freq": 90.0,
        "sampling_freq": 100.0,
        "order": 2
    }