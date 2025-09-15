#!/usr/bin/env python3
"""
Movie Recommendation Assistant (content-based + sentiment-aware)

- Expects a CSV file (default 'imdb_top_1000.csv') with at least the following columns:
  'Series_Title', 'Genre', 'Overview', 'IMDB_Rating'
- Combines Genre + Overview, vectorizes text with TF-IDF, computes cosine similarity.
- Uses TextBlob to analyze mood + movie overview sentiment to provide mood-aligned suggestions.
- Uses Colorama for colored terminal output and a small processing animation.
"""

from textblob import TextBlob
from colorama import init, Fore, Style
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import pandas as pd
import time
import random
import sys
from typing import List, Tuple, Optional

# Initialize colorama
init(autoreset=True)


# -------------------------
# Data loading & preprocessing
# -------------------------
def load_data(file_path: str = "imdb_top_1000.csv") -> pd.DataFrame:
    """
    Loads CSV into pandas DataFrame and creates 'combined_features' column
    by joining Genre and Overview text. Raises on error.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(Fore.RED + f"Error: The file '{file_path}' was not found.")
        sys.exit(1)
    except Exception as e:
        print(Fore.RED + f"Error loading file '{file_path}': {e}")
        sys.exit(1)

    # Ensure expected columns exist
    for col in ("Series_Title", "Genre", "Overview", "IMDB_Rating"):
        if col not in df.columns:
            print(Fore.RED + f"Error: Expected column '{col}' not found in CSV.")
            sys.exit(1)

    # Create combined text field
    df["combined_features"] = df["Genre"].fillna("") + " " + df["Overview"].fillna("")
    # Fill missing IMDB_Rating with NaN and ensure numeric
    df["IMDB_Rating"] = pd.to_numeric(df["IMDB_Rating"], errors="coerce")
    df.reset_index(drop=True, inplace=True)
    return df


# -------------------------
# Vectorization / similarity
# -------------------------
def build_vectorizer_and_similarity(df: pd.DataFrame):
    """
    Builds TF-IDF matrix & cosine similarity matrix for df['combined_features'].
    Returns (tfidf_vectorizer, tfidf_matrix, cosine_sim_matrix)
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(df["combined_features"].fillna(""))
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return tfidf, tfidf_matrix, cosine_sim


# -------------------------
# Utility: list genres
# -------------------------
def list_genres(df: pd.DataFrame) -> List[str]:
    # Split the Genre strings on comma, strip spaces, deduplicate & sort
    genres = sorted(
        {
            genre.strip()
            for entry in df["Genre"].dropna().astype(str)
            for genre in entry.split(",")
            if genre.strip()
        }
    )
    return genres


# -------------------------
# Processing animation (small)
# -------------------------
def processing_animation(steps: int = 3, delay: float = 0.45):
    for _ in range(steps):
        print(Fore.YELLOW + ".", end="", flush=True)
        time.sleep(delay)
    print()  # newline


# -------------------------
# Recommendation logic
# -------------------------
def recommend_movies(
    movies_df: pd.DataFrame,
    cosine_sim,
    genre: Optional[str] = None,
    mood: Optional[str] = None,
    rating: Optional[float] = None,
    top_n: int = 5,
) -> List[Tuple[str, float]]:
    """
    Returns a list of (Series_Title, polarity) for recommended movies.
    Steps:
      - Filter by genre and IMDB rating (if provided)
      - Shuffle for diversity
      - For each candidate, compute sentiment polarity of Overview
      - Keep candidates that align with mood (heuristic from project description)
      - If more candidates than top_n, rank them using a combination of
        polarity and a 'centrality' score from cosine similarity within the filtered set.
    """
    filtered_df = movies_df.copy()

    if genre:
        # case-insensitive contains check on Genre column
        filtered_df = filtered_df[
            filtered_df["Genre"].fillna("").str.contains(genre, case=False, na=False)
        ]

    if rating is not None:
        filtered_df = filtered_df[filtered_df["IMDB_Rating"].notnull()]
        filtered_df = filtered_df[filtered_df["IMDB_Rating"] >= rating]

    if filtered_df.empty:
        return []

    # Shuffle for randomization / diversity
    filtered_df = filtered_df.sample(frac=1, random_state=random.randint(0, 9999)).reset_index(
        drop=True
    )

    # Mood polarity (if provided)
    mood_polarity = None
    if mood:
        mood_polarity = TextBlob(mood).sentiment.polarity

    # Build candidate list with overview polarity
    candidates = []
    for idx, row in filtered_df.iterrows():
        overview = row["Overview"]
        if pd.isna(overview) or str(overview).strip() == "":
            continue
        polarity = TextBlob(str(overview)).sentiment.polarity  # -1 .. +1

        # Mood matching heuristic:
        # - If user mood is negative (polarity < 0): prefer positive or neutral movies
        # - If user mood is positive (polarity > 0): prefer positive or neutral movies
        # - If mood is neutral or not provided: accept any polarity
        matches_mood = True
        if mood_polarity is not None:
            if mood_polarity < 0 and polarity < 0:
                # If user is negative and movie is negative, deprioritize (skip)
                matches_mood = False
            # other cases: allow (we'll rank later)
        if matches_mood:
            candidates.append((idx, row["Series_Title"], polarity))

    if not candidates:
        return []

    # If number of candidates <= top_n: return them (sorted by polarity desc)
    if len(candidates) <= top_n:
        # Sort by polarity descending so more uplifting/positive overviews come first
        sorted_small = sorted(candidates, key=lambda x: x[2], reverse=True)
        return [(title, polarity) for (_, title, polarity) in sorted_small]

    # Otherwise, compute a centrality score using cosine_sim over the filtered set:
    # We'll map filtered_df indexes to global dataframe indexes to use cosine_sim.
    # Create a list of global indices for filtered_df
    global_indices = filtered_df.index.tolist()

    # Create a candidate list of global indices for those we considered
    cand_global_idx = [idx for (idx, _, _) in candidates]

    # For each candidate compute:
    #    avg_similarity = mean(similarity(candidate, other filtered movies))
    # We'll use that as a tie-breaker combined with polarity: score = polarity + 0.5 * avg_similarity
    # Note: cosine_sim is assumed to be a square matrix aligned to movies_df's index order
    scores = []
    for (idx, title, polarity) in candidates:
        # average similarity to other members of filtered_df
        # If cosine_sim is available and shape matches dataframe length, use it; else fallback
        avg_sim = 0.0
        try:
            sims = cosine_sim[idx, filtered_df.index.to_numpy()]
            # remove self (where index equals itself)
            if len(sims) > 1:
                avg_sim = (sims.sum() - 1.0 * (sims.max())) / (len(sims) - 1)
            else:
                avg_sim = sims.mean()
        except Exception:
            avg_sim = 0.0
        combined_score = polarity + 0.6 * avg_sim  # weight similarity moderately
        scores.append((combined_score, title, polarity))

    # sort by combined_score descending and return top_n
    scores.sort(key=lambda x: x[0], reverse=True)
    top = scores[:top_n]
    return [(title, polarity) for (_, title, polarity) in top]


# -------------------------
# Display function
# -------------------------
def display_recommendations(recs: List[Tuple[str, float]], name: str):
    if not recs:
        print(Fore.RED + "No suitable movie recommendations found.\n")
        return
    print(Fore.YELLOW + f"\n🍿 AI-Analyzed Movie Recommendations for {name}:")
    for idx, (title, polarity) in enumerate(recs, start=1):
        sentiment = "Positive 😊" if polarity > 0 else "Negative 😞" if polarity < 0 else "Neutral 😐"
        print(Fore.CYAN + f"{idx}. 🎥 {title} (Polarity: {polarity:.2f}, {sentiment})")
    print()


# -------------------------
# Interactive handler
# -------------------------
def handle_ai(name: str, movies_df: pd.DataFrame, cosine_sim):
    print(Fore.BLUE + "\n🔍 Let's find the perfect movie for you!\n")

    genres = list_genres(movies_df)
    if not genres:
        print(Fore.RED + "No genres available in dataset. Cannot filter by genre.\n")

    else:
        print(Fore.GREEN + "Available Genres:")
        for i, g in enumerate(genres, start=1):
            print(Fore.CYAN + f"{i}. {g}")
        print()

    # Genre selection
    genre = None
    if genres:
        while True:
            genre_input = input(Fore.YELLOW + "Enter genre number or name (or 'skip'): ").strip()
            if genre_input.lower() in ("skip", ""):
                genre = None
                break
            if genre_input.isdigit():
                idx = int(genre_input)
                if 1 <= idx <= len(genres):
                    genre = genres[idx - 1]
                    break
            # try matching by name
            if genre_input.title() in genres:
                genre = genre_input.title()
                break
            print(Fore.RED + "Invalid input. Try again.\n")

    # Mood input
    mood = input(Fore.YELLOW + "How do you feel today? (Describe your mood or 'skip'): ").strip()
    if mood.lower() in ("", "skip"):
        mood = None

    # Analyze mood
    if mood:
        print(Fore.BLUE + "\nAnalyzing mood", end="", flush=True)
        processing_animation()
        mood_p = TextBlob(mood).sentiment.polarity
        mood_desc = "positive 😊" if mood_p > 0 else "negative 😞" if mood_p < 0 else "neutral 😐"
        print(Fore.GREEN + f"Your mood is {mood_desc} (Polarity: {mood_p:.2f}).\n")
    else:
        print(Fore.GREEN + "No mood provided. Proceeding without mood filtering.\n")

    # Rating input
    rating = None
    while True:
        rating_input = input(Fore.YELLOW + "Enter minimum IMDB rating (e.g., 7.0) or 'skip': ").strip()
        if rating_input.lower() in ("skip", ""):
            rating = None
            break
        try:
            rating_val = float(rating_input)
            rating = rating_val
            break
        except ValueError:
            print(Fore.RED + "Invalid input. Try again.\n")

    # Searching animation
    print(Fore.BLUE + f"\nFinding movies for {name}", end="", flush=True)
    processing_animation()

    # Generate recommendations
    recs = recommend_movies(
        movies_df, cosine_sim=cosine_sim, genre=genre, mood=mood, rating=rating, top_n=5
    )

    if isinstance(recs, str):
        print(Fore.RED + recs + "\n")
    else:
        display_recommendations(recs, name)

    # Offer more recommendations
    while True:
        action = input(Fore.YELLOW + "Would you like more recommendations? (yes/no): ").strip().lower()
        if action in ("no", "n"):
            print(Fore.GREEN + f"\nEnjoy your movie picks, {name}! 🎬🍿\n")
            break
        if action in ("yes", "y"):
            print(Fore.BLUE + f"\nFinding more movies for {name}", end="", flush=True)
            processing_animation()
            recs = recommend_movies(movies_df, cosine_sim=cosine_sim, genre=genre, mood=mood, rating=rating, top_n=5)
            display_recommendations(recs, name)
            continue
        print(Fore.RED + "Invalid choice. Try again.\n")


# -------------------------
# main
# -------------------------
def main():
    print(Fore.BLUE + "🎥 Welcome to your Personal Movie Recommendation Assistant! 🎥\n")
    name = input(Fore.YELLOW + "What's your name? ").strip()
    if not name:
        name = "Movie Friend"

    print(Fore.GREEN + f"\nGreat to meet you, {name}!\n")

    # Load data
    movies_df = load_data()  # default filename; change if needed

    # Vectorize & compute similarity
    print(Fore.BLUE + "Preparing dataset", end="", flush=True)
    processing_animation()
    tfidf, tfidf_matrix, cosine_sim = build_vectorizer_and_similarity(movies_df)

    # Hand off to interactive flow
    handle_ai(name, movies_df, cosine_sim)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n" + Fore.RED + "Interrupted by user. Goodbye.")
        sys.exit(0)
