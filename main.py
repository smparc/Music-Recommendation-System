"""
main.py
-------
Entry point for the Music Recommendation System.

Usage
-----
    python main.py --playlist "Kill Bill" "Saturn" "I Hate U" --method both --k 5

Or edit the DEFAULT_PLAYLIST below and run:
    python main.py
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from normalization import z_score_normalize
from svd import svd, explained_variance_ratio
from projection import project_features, interpret_latent_features
from recommend import (
    recommend_one_per_song,
    recommend_overall_top,
    print_recommendations,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "spotify_songs.csv")

FEATURE_COLUMNS = [
    "danceability", "energy", "speechiness", "acousticness",
    "instrumentalness", "liveness", "valence", "loudness", "tempo",
]

DEFAULT_PLAYLIST = [
    "Kill Bill",
    "Saturn",
    "I Hate U",
    "Low",
    "Gone Girl",
]

K = 5  # number of latent dimensions


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at '{path}'.\n"
            "Download the Spotify dataset from:\n"
            "  https://github.com/JulianoOrlandi/Spotify_Top_Songs_and_Audio_Features\n"
            "and place it at data/spotify_songs.csv"
        )
    df = pd.read_csv(path)
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[FEATURE_COLUMNS].to_numpy(dtype=float)


def find_playlist_indices(df: pd.DataFrame, song_titles: list[str]) -> list[int]:
    indices = []
    for title in song_titles:
        mask = df["track_name"].str.lower() == title.lower()
        matches = df[mask].index.tolist()
        if matches:
            indices.append(matches[0])
            print(f"  ✓ Found: '{title}'")
        else:
            print(f"  ✗ Not found: '{title}' — skipping")
    return indices


def print_latent_features(interpretations: dict, sigma: np.ndarray):
    evr = explained_variance_ratio(sigma)
    print("\n── Latent Feature Analysis ──────────────────────────────")
    for i, pairs in interpretations.items():
        print(f"\n  Latent Feature {i}  (σ={sigma[i-1]:.2f}, var={evr[i-1]*100:.1f}%)")
        terms = []
        for weight, name in pairs[:5]:  # top 5 contributors
            sign = "+" if weight >= 0 else "−"
            terms.append(f"  {sign}{abs(weight):.3f}×{name}")
        print("  " + "  ".join(terms))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Music Recommendation via SVD")
    parser.add_argument("--playlist", nargs="+", default=DEFAULT_PLAYLIST,
                        help="Song titles to use as the input playlist")
    parser.add_argument("--method", choices=["one_per_song", "overall_top", "both"],
                        default="both", help="Recommendation strategy")
    parser.add_argument("--k", type=int, default=K,
                        help="Number of latent dimensions")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of recommendations (default = playlist length)")
    args = parser.parse_args()

    # 1. Load data
    print("\n── Loading dataset ──────────────────────────────────────")
    df = load_dataset(DATA_PATH)
    print(f"  Loaded {len(df)} songs with {len(FEATURE_COLUMNS)} audio features")

    # 2. Build & normalize feature matrix
    B_raw = build_feature_matrix(df)
    B_norm, mean, std = z_score_normalize(B_raw)
    print(f"  Feature matrix shape: {B_norm.shape}")

    # 3. Project into latent space
    print(f"\n── SVD Projection (k={args.k}) ─────────────────────────────")
    B_k, Vk, sigma, U = project_features(B_norm, k=args.k)
    evr = explained_variance_ratio(sigma)
    print(f"  Singular values: {sigma.round(2)}")
    print(f"  Variance explained: {(evr * 100).round(1)}%  (total: {evr.sum()*100:.1f}%)")

    # 4. Print latent feature interpretations
    interp = interpret_latent_features(Vk, FEATURE_COLUMNS)
    print_latent_features(interp, sigma)

    # 5. Find playlist songs
    print(f"\n── Input Playlist ({len(args.playlist)} songs) ──────────────────────")
    playlist_indices = find_playlist_indices(df, args.playlist)
    if not playlist_indices:
        print("No playlist songs found in dataset. Exiting.")
        return

    # 6. Recommend
    n_recs = args.n if args.n else len(playlist_indices)
    song_metadata = df.to_dict(orient="records")

    if args.method in ("one_per_song", "both"):
        recs = recommend_one_per_song(playlist_indices, B_k, song_metadata)
        print_recommendations(recs, title="One-Per-Song Recommendations")

    if args.method in ("overall_top", "both"):
        recs = recommend_overall_top(playlist_indices, B_k, song_metadata, n_recommendations=n_recs)
        print_recommendations(recs, title="Overall Top Recommendations")


if __name__ == "__main__":
    main()
