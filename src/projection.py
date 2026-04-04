"""
projection.py
-------------
Project song feature vectors into a reduced latent space using SVD,
then compute cosine similarity between songs in that space.

Mathematical foundation:
    B_k = B · V_k

This is equivalent to projecting each song row onto the orthonormal
basis defined by the top-k right singular vectors.

Cosine similarity between two vectors a, b:
    cos(a, b) = (a · b) / (||a|| · ||b||)
"""

import numpy as np
from svd import svd, explained_variance_ratio


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

def project_features(B: np.ndarray, k: int = 5):
    """
    Project feature matrix B into a k-dimensional latent space via SVD.

    Parameters
    ----------
    B : np.ndarray (n_songs, n_features) — normalized feature matrix
    k : int — number of latent dimensions to keep

    Returns
    -------
    B_k    : np.ndarray (n_songs, k)   — projected song representations
    Vk     : np.ndarray (n_features, k) — projection basis (columns of V)
    sigma  : np.ndarray (k,)            — singular values
    U      : np.ndarray (n_songs, k)    — left singular vectors
    """
    U, sigma, Vt = svd(B, k=k)
    Vk = Vt.T  # (n_features, k)
    B_k = B @ Vk
    return B_k, Vk, sigma, U


def project_new_song(x: np.ndarray, Vk: np.ndarray) -> np.ndarray:
    """
    Project a new (normalized) song feature vector into the latent space.

    Parameters
    ----------
    x  : np.ndarray (n_features,) — normalized feature vector
    Vk : np.ndarray (n_features, k) — projection basis

    Returns
    -------
    np.ndarray (k,) — projected song vector
    """
    return x @ Vk


def interpret_latent_features(Vk: np.ndarray, feature_names: list) -> dict:
    """
    Return a human-readable description of each latent feature as a
    weighted linear combination of the original features.

    Parameters
    ----------
    Vk           : np.ndarray (n_features, k)
    feature_names: list of str

    Returns
    -------
    dict: {latent_feature_index: [(weight, feature_name), ...]}
    """
    k = Vk.shape[1]
    interpretations = {}
    for i in range(k):
        weights = Vk[:, i]
        pairs = sorted(zip(weights, feature_names), key=lambda x: abs(x[0]), reverse=True)
        interpretations[i + 1] = pairs
    return interpretations


# ---------------------------------------------------------------------------
# Cosine Similarity
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.

    cos(a, b) = dot(a, b) / (norm(a) * norm(b))
    """
    norm_a = np.sqrt(np.dot(a, a))
    norm_b = np.sqrt(np.dot(b, b))
    if norm_a < 1e-10 or norm_b < 1e-10:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_matrix(B_k: np.ndarray) -> np.ndarray:
    """
    Compute the full pairwise cosine similarity matrix for all songs.

    Parameters
    ----------
    B_k : np.ndarray (n_songs, k) — latent song representations

    Returns
    -------
    sim : np.ndarray (n_songs, n_songs) — pairwise cosine similarities
    """
    # Normalize each row
    norms = np.sqrt(np.sum(B_k ** 2, axis=1, keepdims=True))
    norms = np.where(norms < 1e-10, 1.0, norms)
    B_norm = B_k / norms

    sim = B_norm @ B_norm.T
    return sim


def song_similarities(query_vec: np.ndarray, B_k: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between a single query song and all database songs.

    Parameters
    ----------
    query_vec : np.ndarray (k,)
    B_k       : np.ndarray (n_songs, k)

    Returns
    -------
    np.ndarray (n_songs,) — similarity scores
    """
    scores = np.array([cosine_similarity(query_vec, B_k[i]) for i in range(B_k.shape[0])])
    return scores
