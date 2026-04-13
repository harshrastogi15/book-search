# 📚 Semantic Book Search Engine

A semantic book search engine built with Sentence-BERT embeddings and FastAPI.
Unlike keyword search, this searches by **meaning** — not just exact words.

## 🔍 Example
Search: `"a boy who discovers magical powers"`
Returns: `A Wizard of Earthsea`, `Hidden Talents` — without those exact words in the query.

## 🛠️ Tech Stack
- **Sentence-BERT** — generates text embeddings
- **FAISS** — fast vector similarity search
- **FastAPI** — REST API framework
- **Streamlit** — interactive web UI
- **Pandas / NumPy** — data processing

## 📁 Project Structure

```
book-search/
├── data/                      # dataset (not pushed to GitHub)
├── embeddings/                # saved vectors (not pushed to GitHub)
├── notebooks/                 # data exploration
├── main.py                    # FastAPI app
├── search_cosine.py           # cosine search logic
├── search_faiss.py            # faiss search logic
├── generate_embeddings.py     # embedding generation script
├── app.py                     # Streamlit UI
└── requirements.txt           # dependencies
```
## 🚀 Run Locally

### 1. Clone the repo
```bash
git clone https://github.com/YOURUSERNAME/book-search.git
cd book-search
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download dataset
Download the [7k Books Dataset](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata)
and place `books.csv` in the `data/` folder.

### 5. Generate embeddings
```bash
python generate_embeddings.py
```

### 6. Run the API
```bash
uvicorn main:app --reload
```
API runs at `http://localhost:8000`
Docs at `http://localhost:8000/docs`

### 7. Run the UI (optional)
```bash
streamlit run app.py
```

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Health check |
| GET | `/search` | Search books semantically |
| GET | `/categories` | Get all categories |

## 📊 How It Works
1. Book descriptions are converted to 384-dimensional vectors using Sentence-BERT
2. User query is embedded the same way
3. Cosine similarity finds the closest book vectors
4. Results are ranked by similarity score and returned as JSON

## 🧠 Technical Decisions & Why

### Why Sentence-BERT over other embedding models?

Sentence-BERT (all-MiniLM-L6-v2) was chosen because it is specifically 
trained to produce meaningful sentence-level embeddings, unlike plain BERT 
which works better at word level. It is also lightweight and fast while 
maintaining high accuracy — making it practical for a dataset of 5000+ books 
without needing a GPU.

Alternatives considered:
- **Plain BERT** — not optimized for sentence similarity tasks
- **OpenAI embeddings** — better quality but requires paid API key
- **Word2Vec** — only works at word level, misses sentence context

---

### Why Cosine Similarity over other distance metrics?

Cosine similarity measures the **angle** between two vectors rather than 
the distance between them. This makes it ideal for text embeddings because:

- A short book description and a long book description can mean the same 
  thing but have very different vector magnitudes
- Cosine similarity ignores magnitude and only compares direction
- Two vectors pointing in the same direction = similar meaning, 
  regardless of their size

Alternatives considered:
- **Euclidean Distance** — measures straight line distance between vectors.
  Bad for text because it is heavily affected by vector magnitude, not just 
  direction. A short and long description of the same topic would appear 
  very different.
- **Dot Product** — similar to cosine but not normalized, so longer texts 
  would always score higher regardless of meaning match.
- **Manhattan Distance** — even more sensitive to magnitude than Euclidean, 
  not suitable for high dimensional text embeddings.

---

## ⚡ Why FAISS for Vector Search?

### The Problem With Brute Force Search

When a user searches for a book, we need to compare their query vector
against every single book vector in our dataset. This is called
brute force search. It works fine for small datasets but does not scale.

| Books       | Brute Force  | FAISS        |
|-------------|--------------|--------------|
| 5,000       | 0.003s ✅    | 0.001s ✅    |
| 100,000     | 0.05s ✅     | 0.001s ✅    |
| 1,000,000   | 0.5s ⚠️      | 0.002s ✅    |
| 10,000,000  | 5s ❌        | 0.003s ✅    |

Brute force time grows linearly as dataset grows.
FAISS stays flat regardless of how many books you add.

---

### What is FAISS?

FAISS (Facebook AI Similarity Search) is a library built by Meta that
specializes in fast similarity search over large collections of vectors.

Instead of checking every vector one by one, FAISS builds a smart
index structure upfront that organizes vectors by similarity.
Search then navigates this structure directly to the answer — skipping
irrelevant vectors entirely. Think of it like a librarian who
organizes books by topic before you even walk in, so finding
what you need takes seconds instead of hours.

---

### Why IndexFlatIP Specifically?

FAISS has many index types for different use cases:

| Index          | Speed    | Accuracy | Best For           |
|----------------|----------|----------|--------------------|
| IndexFlatL2    | Slow     | 100%     | Euclidean distance |
| IndexFlatIP    | Slow     | 100%     | Cosine similarity  |
| IndexIVFFlat   | Fast     | ~99%     | 100k+ vectors      |
| IndexIVFPQ     | Fastest  | ~95%     | 1M+ vectors        |
| IndexHNSWFlat  | Fastest  | ~99%     | Production systems |

We use IndexFlatIP because our dataset is small (5729 books) so exact
results matter more than raw speed. IP (Inner Product) on normalized
vectors is mathematically equal to cosine similarity — so we get
familiar cosine behavior with FAISS efficiency.

For 1 million+ books, switching to IndexIVFFlat or IndexHNSWFlat
would give dramatic speed improvements with minimal accuracy loss.

---