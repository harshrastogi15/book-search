import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

print("Loading data and model (FAISS)...")
df = pd.read_csv('data/cleaned_books.csv')
model = SentenceTransformer('all-MiniLM-L6-v2')

# load saved FAISS index
print("Loading FAISS index...")
index = faiss.read_index('embeddings/books.index')
print(f"✅ FAISS search ready! {index.ntotal} books indexed.")


def search_books(query, top_k=5, min_rating=0.0, category=None):
    """
    FAISS powered semantic search.
    Uses IndexFlatIP (Inner Product on normalized vectors = cosine similarity).
    Stays fast even as dataset grows to millions.
    """

    # embed and normalize query
    query_vector = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)

    # fetch more than top_k to account for filtered results
    fetch_k = min(top_k * 10, index.ntotal)

    # FAISS search — returns scores and indices
    scores, indices = index.search(query_vector, fetch_k)

    # build results
    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]

    # apply filters
    if min_rating > 0.0:
        results = results[results['average_rating'] >= min_rating]

    if category:
        results = results[
            results['categories'].str.contains(category, case=False, na=False)
        ]

    return results.head(top_k)[[
        'title',
        'authors',
        'categories',
        'average_rating',
        'description',
        'thumbnail',
        'similarity_score'
    ]]


def get_all_categories():
    cats = df['categories'].dropna().unique().tolist()
    return sorted(set(cats))