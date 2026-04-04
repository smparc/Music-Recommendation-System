"""
normalization.py
----------------
Z-score normalization for audio feature matrices.
All math is implemented from scratch using only NumPy.
"""

import numpy as np


def compute_mean(X: np.ndarray) -> np.ndarray:
    """Compute the column-wise mean of matrix X."""
    return np.sum(X, axis=0) / X.shape[0]


def compute_std(X: np.ndarray, mean: np.ndarray) -> np.ndarray:
    """Compute the column-wise standard deviation of matrix X."""
    diff = X - mean
    variance = np.sum(diff ** 2, axis=0) / X.shape[0]
    return np.sqrt(variance)


def z_score_normalize(X: np.ndarray):
    """
    Normalize a feature matrix using z-score normalization.

    z = (x - mu) / sigma

    Parameters
    ----------
    X : np.ndarray of shape (n_songs, n_features)

    Returns
    -------
    X_norm : np.ndarray
        Normalized matrix.
    mean : np.ndarray
        Column means (for later inverse transform or new-point normalization).
    std : np.ndarray
        Column standard deviations.
    """
    mean = compute_mean(X)
    std = compute_std(X, mean)

    # Avoid division by zero for constant features
    std_safe = np.where(std == 0, 1.0, std)
    X_norm = (X - mean) / std_safe

    return X_norm, mean, std_safe


def normalize_new_point(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply previously computed mean/std to a new song feature vector.

    Parameters
    ----------
    x : np.ndarray of shape (n_features,)
    mean, std : previously computed normalization statistics

    Returns
    -------
    np.ndarray — normalized feature vector
    """
    return (x - mean) / std
