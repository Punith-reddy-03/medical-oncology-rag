"""
=============================================================
HYBRID RETRIEVAL FOR ONCOLOGY RAG
- Dense + Sparse (BM25) + Reranking
- Optimized for higher Precision@5 and Recall@5
=============================================================
"""

import os
import torch
import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# =========================================
# CONFIGURATION
# =========================================

CHROMA_PATH     = "./chroma_store"
COLLECTION_NAME = "medical_books"
TOP_K           = 15          # Retrieve more for better recall
FINAL_K         = 5           # Final top-k after reranking
BM25_WEIGHT     = 0.35
VECTOR_WEIGHT   = 0.35
RERANKER_WEIGHT = 0.30
USE_CROSS_ENCODER = True      # Better relevance scoring

# =========================================
# DEVICE DETECTION
# =========================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

print(f"✅ Device: {DEVICE.upper()}\n")

# Set CPU threads for better performance
torch.set_num_threads(os.cpu_count())
print(f"✅ Using {os.cpu_count()} CPU threads\n")

# =========================================
# LOAD MODELS
# =========================================

print("Loading embedding model (all-MiniLM-L6-v2)...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)
print("✅ Embedding model loaded\n")

# Load Cross-Encoder for reranking (better relevance scoring)
if USE_CROSS_ENCODER:
    print("Loading Cross-Encoder (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
    try:
        reranker = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            device=DEVICE,
            max_length=512
        )
        print("✅ Cross-Encoder loaded\n")
    except Exception as e:
        print(f"⚠️ Could not load Cross-Encoder: {e}")
        print("   Using similarity-based reranking only\n")
        USE_CROSS_ENCODER = False

# =========================================
# CONNECT TO CHROMADB
# =========================================

print(f"Connecting to ChromaDB at '{CHROMA_PATH}'...")
client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = client.get_collection(name=COLLECTION_NAME)
print(f"✅ Connected! Total chunks in DB: {collection.count()}\n")


# =========================================
# STOPWORDS FOR BM25
# =========================================

try:
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'of', 'to', 'for', 'in', 
                  'on', 'at', 'by', 'with', 'without', 'is', 'are', 'was', 
                  'were', 'be', 'been', 'being', 'have', 'has', 'had', 
                  'having', 'do', 'does', 'did', 'doing', 'but', 'so', 
                  'if', 'then', 'else', 'when', 'where', 'which', 'what',
                  'who', 'whom', 'this', 'that', 'these', 'those'}


def preprocess_text(text: str) -> list:
    """Preprocess text for BM25: lowercasing, tokenization, stopword removal."""
    tokens = word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalnum() and t not in STOP_WORDS and len(t) > 1]
    return tokens


# =========================================
# RETRIEVAL FUNCTIONS
# =========================================

def get_all_chunks_by_category(category: str):
    """Get all chunks for a specific category."""
    try:
        results = collection.get(
            where={"category": category},
            include=["documents", "embeddings", "metadatas"]
        )
        return results
    except Exception as e:
        print(f"  Error getting chunks by category: {e}")
        return {"documents": [], "embeddings": [], "metadatas": [], "ids": []}


def bm25_search(query: str, documents: list):
    """BM25 search with improved preprocessing."""
    # Preprocess all documents
    tokenized_docs = [preprocess_text(doc) for doc in documents]
    
    # Preprocess query
    tokenized_query = preprocess_text(query)
    
    if not tokenized_docs or not tokenized_query:
        return np.zeros(len(documents))
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_docs)
    
    # Get scores
    scores = bm25.get_scores(tokenized_query)
    
    return scores


def vector_search(query: str, category: str, top_k: int):
    """Vector similarity search with optimized embeddings."""
    # Encode query
    query_embedding = embed_model.encode(
        [query],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()
    
    # Search with category filter
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"category": category},
        include=["documents", "embeddings", "metadatas", "distances"]
    )
    
    return results


def cross_encoder_rerank(query: str, documents: list, scores: list) -> list:
    """Rerank documents using Cross-Encoder for better relevance."""
    if not documents or not USE_CROSS_ENCODER:
        return scores
    
    # Prepare pairs for cross-encoder
    pairs = [[query, doc[:512]] for doc in documents]
    
    try:
        # Get cross-encoder scores
        ce_scores = reranker.predict(pairs)
        # Normalize to [0, 1]
        ce_scores = (ce_scores - ce_scores.min()) / (ce_scores.max() - ce_scores.min() + 1e-8)
        
        # Combine with original scores
        combined = [RERANKER_WEIGHT * ce + (1 - RERANKER_WEIGHT) * s 
                   for ce, s in zip(ce_scores, scores)]
        return combined
    except Exception as e:
        print(f"  Cross-encoder reranking failed: {e}")
        return scores


def hybrid_search(query: str, category: str, top_k: int = TOP_K, final_k: int = FINAL_K) -> list:
    """
    Enhanced hybrid search with:
    - BM25 for sparse retrieval
    - Vector for dense retrieval  
    - Cross-encoder for reranking
    """
    print(f"\nRunning enhanced hybrid search...")
    print(f"Query    : {query[:80]}...")
    print(f"Category : {category}")
    print(f"Top-K    : {top_k} (rerank to {final_k})\n")
    
    # --- Get all chunks in category ---
    category_data = get_all_chunks_by_category(category)
    
    if not category_data["documents"]:
        print(f"⚠️ No chunks found for category: '{category}'")
        return []
    
    documents = category_data["documents"]
    metadatas = category_data["metadatas"]
    ids = category_data["ids"]
    
    print(f"Total chunks in category '{category}': {len(documents)}")
    
    # --- BM25 Scores ---
    print("  Computing BM25 scores...")
    bm25_scores = bm25_search(query, documents)
    
    # Normalize BM25 scores
    bm25_min = bm25_scores.min()
    bm25_max = bm25_scores.max()
    if bm25_max - bm25_min > 1e-6:
        bm25_scores_norm = (bm25_scores - bm25_min) / (bm25_max - bm25_min)
    else:
        bm25_scores_norm = np.zeros_like(bm25_scores)
    
    # --- Vector Scores ---
    print("  Computing vector similarity scores...")
    # Get top_k*2 from vector search for better coverage
    vector_results = vector_search(query, category, top_k * 2)
    
    vector_id_score = {}
    if vector_results["ids"][0]:
        for vid, dist in zip(vector_results["ids"][0], vector_results["distances"][0]):
            vector_score = 1 - dist  # Convert distance to similarity
            vector_id_score[vid] = vector_score
    
    vector_scores = np.zeros(len(ids))
    for i, chunk_id in enumerate(ids):
        if chunk_id in vector_id_score:
            vector_scores[i] = vector_id_score[chunk_id]
    
    # Normalize vector scores
    vec_min = vector_scores.min()
    vec_max = vector_scores.max()
    if vec_max - vec_min > 1e-6:
        vector_scores_norm = (vector_scores - vec_min) / (vec_max - vec_min)
    else:
        vector_scores_norm = np.zeros_like(vector_scores)
    
    # --- Initial Hybrid Scores ---
    hybrid_scores = (BM25_WEIGHT * bm25_scores_norm) + (VECTOR_WEIGHT * vector_scores_norm)
    
    # --- Cross-Encoder Reranking ---
    if USE_CROSS_ENCODER:
        print("  Reranking with Cross-Encoder...")
        # Get top candidates for reranking
        top_indices = np.argsort(hybrid_scores)[::-1][:top_k]
        top_docs = [documents[i] for i in top_indices]
        top_scores = [hybrid_scores[i] for i in top_indices]
        
        # Rerank
        reranked_scores = cross_encoder_rerank(query, top_docs, top_scores)
        
        # Update scores
        for idx, (orig_idx, new_score) in enumerate(zip(top_indices, reranked_scores)):
            hybrid_scores[orig_idx] = new_score
    
    # --- Final Ranking ---
    ranked_indices = np.argsort(hybrid_scores)[::-1][:final_k]
    
    final_results = []
    for rank, idx in enumerate(ranked_indices):
        final_results.append({
            "rank": rank + 1,
            "id": ids[idx],
            "content": documents[idx],
            "metadata": metadatas[idx],
            "bm25_score": round(float(bm25_scores_norm[idx]), 4),
            "vector_score": round(float(vector_scores_norm[idx]), 4),
            "hybrid_score": round(float(hybrid_scores[idx]), 4),
        })
    
    print(f"\n✅ Retrieved {len(final_results)} chunks\n")
    return final_results


def list_categories():
    """List all available categories in the database."""
    all_meta = collection.get(include=["metadatas"])["metadatas"]
    cats = {}
    for m in all_meta:
        cat = m.get("category", "Unknown")
        cats[cat] = cats.get(cat, 0) + 1
    print("\n--- Available Categories ---")
    for cat, count in sorted(cats.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cat:<25} : {count} chunks")
    print()
    return list(cats.keys())


if __name__ == "__main__":
    # Test
    list_categories()
    
    # Test query
    test_query = "What are the treatment options for lung cancer?"
    test_category = "Treatment"
    
    results = hybrid_search(test_query, test_category)
    
    print("\n" + "=" * 60)
    print("SEARCH RESULTS")
    print("=" * 60)
    for r in results:
        print(f"\nRank {r['rank']}: {r['metadata'].get('source', 'Unknown')}")
        print(f"  Score: {r['hybrid_score']}")
        print(f"  Content: {r['content'][:200]}...")
        print("-" * 40)