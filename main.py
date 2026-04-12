from fastapi import FastAPI, Query
from search import search_books, get_all_categories
from typing import Optional

app = FastAPI(
    title="📚 Book Search API",
    description="Semantic book search using Sentence-BERT embeddings",
    version="1.0.0"
)


# ---- Health Check ----
@app.get("/")
def root():
    return {
        "message": "Book Search API is running!",
        "endpoints": {
            "search": "/search",
            "categories": "/categories",
            "docs": "/docs"
        }
    }


# ---- Search Endpoint ----
@app.get("/search")
def search(
    query: str = Query(..., description="What kind of book are you looking for?"),
    top_k: int = Query(5, description="Number of results to return"),
    min_rating: float = Query(0.0, description="Minimum average rating"),
    category: Optional[str] = Query(None, description="Filter by category")
):
    """
    Search books semantically based on meaning of your query.
    """

    results = search_books(
        query=query,
        top_k=top_k,
        min_rating=min_rating,
        category=category
    )

    if results.empty:
        return {
            "query": query,
            "total_results": 0,
            "results": []
        }

    return {
        "query": query,
        "total_results": len(results),
        "results": results.to_dict(orient='records')
    }


# ---- Categories Endpoint ----
@app.get("/categories")
def categories():
    """
    Returns all available book categories for filtering.
    """
    return {
        "total": len(get_all_categories()),
        "categories": get_all_categories()
    }