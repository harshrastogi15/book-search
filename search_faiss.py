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

    query_vector = model.encode([query]).astype('float32')
    faiss.normalize_L2(query_vector)

    fetch_k = min(top_k * 10, index.ntotal)

    scores, indices = index.search(query_vector, fetch_k)

    results = df.iloc[indices[0]].copy()
    results['similarity_score'] = scores[0]

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