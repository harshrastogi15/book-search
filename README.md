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
├── search.py                  # core search logic
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