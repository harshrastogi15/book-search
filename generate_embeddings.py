import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import time

# ---- Load cleaned data ----
print("Loading cleaned data...")
df = pd.read_csv('data/cleaned_books.csv')
print(f"Total books: {len(df)}")

# ---- Load pre-trained model ----
print("\nLoading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("Model loaded!")

# ---- Generate embeddings ----
print("\nGenerating embeddings... (this may take 2-3 minutes)")
start_time = time.time()

embeddings = model.encode(
    df['text_to_embed'].tolist(),
    show_progress_bar=True,    # shows a nice progress bar
    batch_size=64              # processes 64 books at a time
)

end_time = time.time()
print(f"\nDone! Took {round(end_time - start_time, 2)} seconds")

# ---- Check shape ----
print(f"Embeddings shape: {embeddings.shape}")

# ---- Save embeddings ----
print("\nSaving embeddings...")
np.save('embeddings/book_embeddings.npy', embeddings)
df.to_csv('data/cleaned_books.csv', index=False)
print("✅ Embeddings saved to embeddings/book_embeddings.npy")