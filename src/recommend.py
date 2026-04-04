"""
recommend.py
------------
Two recommendation strategies built on top of SVD-projected latent features:

1. One-Per-Song  — prioritizes diversity; one recommendation per input song
2. Overall Top   — prioritizes coherence; top globally-similar songs overall
"""

import numpy as np
from projection import song_similarities


# ---------------------------------------------------------------------------
# Strategy 1: One-Per-Song
# ---------------------------------------------------------------------------

def recommend_one_per_song(
    playlist_indices: list[int],
    B_k: np.ndarray,
    song_metadata: list[dict],
    exclude_indices: set = None,
) -> list[dict]:
    """
    For each song in the input playlist, find its single most similar
    song in the database that hasn't already been recommended.

    Parameters
    ----------
    playlist_indices : list[int]
        Indices (into B_k / song_metadata) of the input playlist songs.
    B_k : np.ndarray (n_songs, k)
        Latent-space representations of all songs.
    song_metadata : list[dict]
        Metadata for each song (keys: 'track_name', 'artist_names', etc.)
    exclude_indices : set, optional
        Indices to exclude from recommendations (e.g., input playlist itself).

    Returns
    -------
    list[dict] — one recommended song per input song, with similarity score
    """
    if exclude_indices is None:
        exclude_indices = set(playlist_indices)
    else:
        exclude_indices = set(exclude_indices) | set(playlist_indices)

    recommendations = []
    used_indices = set()

    for idx in playlist_indices:
        query_vec = B_k[idx]
        scores = song_similarities(query_vec, B_k)

        # Sort by similarity descending
        sorted_indices = np.argsort(scores)[::-1]

        for candidate_idx in sorted_indices:
            candidate_idx = int(candidate_idx)
            if candidate_idx in exclude_indices:
                continue
            if candidate_idx in used_indices:
                continue

            rec = dict(song_metadata[candidate_idx])
            rec["similarity_score"] = float(scores[candidate_idx])
            rec["recommended_for"] = song_metadata[idx].get("track_name", str(idx))
            recommendations.append(rec)
            used_indices.add(candidate_idx)
            break  # one per input song

    return recommendations


# ---------------------------------------------------------------------------
# Strategy 2: Overall Top
# ---------------------------------------------------------------------------

def recommend_overall_top(
    playlist_indices: list[int],
    B_k: np.ndarray,
    song_metadata: list[dict],
    n_recommendations: int = None,
    exclude_indices: set = None,
) -> list[dict]:
    """
    Collect similarity scores from all input playlist songs vs. the full
    database, then return the globally highest-scoring candidates.

    Parameters
    ----------
    playlist_indices : list[int]
    B_k : np.ndarray (n_songs, k)
    song_metadata : list[dict]
    n_recommendations : int
        How many songs to return. Defaults to len(playlist_indices).
    exclude_indices : set, optional

    Returns
    -------
    list[dict] — top-N globally most similar songs
    """
    if n_recommendations is None:
        n_recommendations = len(playlist_indices)

    if exclude_indices is None:
        exclude_indices = set(playlist_indices)
    else:
        exclude_indices = set(exclude_indices) | set(playlist_indices)

    n_songs = B_k.shape[0]
    # Aggregate similarity: max similarity across all input songs
    agg_scores = np.full(n_songs, -np.inf)

    for idx in playlist_indices:
        query_vec = B_k[idx]
        scores = song_similarities(query_vec, B_k)
        agg_scores = np.maximum(agg_scores, scores)

    # Zero out excluded songs
    for idx in exclude_indices:
        agg_scores[idx] = -np.inf

    sorted_indices = np.argsort(agg_scores)[::-1]

    recommendations = []
    seen = set()
    for candidate_idx in sorted_indices:
        candidate_idx = int(candidate_idx)
        if candidate_idx in exclude_indices:
            continue
        if candidate_idx in seen:
            continue
        if len(recommendations) >= n_recommendations:
            break

        rec = dict(song_metadata[candidate_idx])
        rec["similarity_score"] = float(agg_scores[candidate_idx])
        recommendations.append(rec)
        seen.add(candidate_idx)

    return recommendations


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def print_recommendations(recommendations: list[dict], title: str = "Recommendations"):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")
    for i, rec in enumerate(recommendations, 1):
        track = rec.get("track_name", "Unknown")
        artist = rec.get("artist_names", "Unknown")
        score = rec.get("similarity_score", 0.0)
        print(f"  {i:>2}. {track} — {artist}  [sim={score:.4f}]")
    print(f"{'=' * 60}\n")
