import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import time

print("Loading cleaned data...")
df = pd.read_csv('data/cleaned_books.csv')
print(f"Total books: {len(df)}")

print("\nLoading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

print("\nGenerating embeddings...")
start = time.time()

embeddings = model.encode(
    df['text_to_embed'].tolist(),
    show_progress_bar=True,
    batch_size=64
)

print(f"Done! Took {round(time.time() - start, 2)} seconds")
print(f"Embeddings shape: {embeddings.shape}")

print("\nSaving raw embeddings...")
np.save('embeddings/book_embeddings.npy', embeddings)
print("✅ Saved to embeddings/book_embeddings.npy")

print("\nBuilding FAISS index...")
embeddings_f32 = embeddings.astype('float32')
faiss.normalize_L2(embeddings_f32)

dimension = embeddings_f32.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_f32)

faiss.write_index(index, 'embeddings/books.index')
print(f"✅ FAISS index saved to embeddings/books.index")
print(f"✅ Total books indexed: {index.ntotal}")