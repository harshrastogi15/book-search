import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---- Load everything once when this file is imported ----
print("Loading data and model...")
df = pd.read_csv('data/cleaned_books.csv')
embeddings = np.load('embeddings/book_embeddings.npy')
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Ready!")


def search_books(query, top_k=5, min_rating=0.0, category=None):
    """
    Search books by meaning/semantic similarity.

    query      : what the user types
    top_k      : how many results to return
    min_rating : filter by minimum average rating
    category   : filter by book category
    """

    # Step 1 — embed the user query into a vector
    query_vector = model.encode([query])

    # Step 2 — compare query vector against all book vectors
    similarities = cosine_similarity(query_vector, embeddings)[0]

    # Step 3 — sort by similarity, highest first
    top_indices = np.argsort(similarities)[::-1]

    # Step 4 — build results dataframe
    results = df.iloc[top_indices].copy()
    results['similarity_score'] = similarities[top_indices]

    # Step 5 — apply filters
    if min_rating > 0.0:
        results = results[results['average_rating'] >= min_rating]

    if category:
        results = results[
            results['categories'].str.contains(category, case=False, na=False)
        ]

    # Step 6 — return top k results
    results = results.head(top_k)

    return results[[
        'title',
        'authors',
        'categories',
        'average_rating',
        'description',
        'thumbnail',
        'similarity_score'
    ]]


def get_all_categories():
    """Returns a sorted list of unique categories for the UI filter."""
    cats = df['categories'].dropna().unique().tolist()
    cats = sorted(set(cats))
    return cats