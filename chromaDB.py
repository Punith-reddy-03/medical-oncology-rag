import pickle
import os
import torch
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# =========================================
# CONFIGURATION
# =========================================

CHUNKS_FILE     = "semantic_chunks.pkl"
CHROMA_PATH     = "./chroma_store"
COLLECTION_NAME = "medical_books"
BATCH_SIZE      = 200   # Reduced from 500 → 200 for CPU/low RAM


# =========================================
# AUTO DETECT DEVICE (Intel Iris Xe → CPU)
# =========================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
    print("✅ Mac M2 GPU (MPS) detected — using MPS!\n")
elif torch.cuda.is_available():
    DEVICE = "cuda"
    print("✅ NVIDIA GPU detected — using CUDA!\n")
else:
    DEVICE = "cpu"
    print("⚠️ Intel Iris Xe Graphics detected — using optimized CPU mode\n")


# =========================================
# INTEL EXTENSION FOR PYTORCH (IPEX)
# =========================================

IPEX_AVAILABLE = False
if DEVICE == "cpu":
    try:
        import intel_extension_for_pytorch as ipex
        IPEX_AVAILABLE = True
        print("✅ Intel Extension for PyTorch (IPEX) loaded!\n")
    except ImportError:
        print("ℹ️  IPEX not installed. Running standard CPU mode.\n")

# Use all CPU cores
torch.set_num_threads(os.cpu_count())
print(f"✅ Using {os.cpu_count()} CPU threads\n")


# =========================================
# LOAD SEMANTIC CHUNKS
# =========================================

print(f"Loading {CHUNKS_FILE}...")

if not os.path.exists(CHUNKS_FILE):
    print(f"❌ ERROR: '{CHUNKS_FILE}' not found!")
    print("   Make sure semantic_chunks.pkl is in the same folder.")
    exit(1)

with open(CHUNKS_FILE, "rb") as f:
    semantic_chunks = pickle.load(f)

print(f"✅ Loaded {len(semantic_chunks)} chunks\n")


# =========================================
# LOAD EMBEDDING MODEL
# =========================================

print("Loading embedding model...")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

# Apply IPEX optimization if available
if IPEX_AVAILABLE:
    try:
        model[0].auto_model = ipex.optimize(model[0].auto_model)
        print("✅ Model optimized with IPEX!\n")
    except Exception as e:
        print(f"ℹ️  IPEX optimization skipped: {e}\n")

print("✅ Model loaded\n")


# =========================================
# SETUP CHROMADB
# =========================================

print(f"Setting up ChromaDB at '{CHROMA_PATH}'...")

client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)

# Uncomment below to reset/clear collection if needed:
# client.delete_collection(COLLECTION_NAME)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

print(f"✅ Collection '{COLLECTION_NAME}' ready\n")


# =========================================
# CHECK ALREADY STORED (RESUME SUPPORT)
# =========================================

already_stored = collection.count()
if already_stored > 0:
    print(f"ℹ️  Found {already_stored} chunks already in ChromaDB.")
    print(f"   Will skip already stored chunks and continue from where left off.\n")

start_index = already_stored  # Resume from last saved position


# =========================================
# UPSERT CHUNKS INTO CHROMADB
# =========================================

total = len(semantic_chunks)
remaining = total - start_index

if remaining <= 0:
    print("✅ All chunks already stored in ChromaDB! Nothing to do.\n")
else:
    print(f"Starting upsert of {remaining} remaining chunks")
    print(f"(Skipping first {start_index} already stored)\n")

    for batch_start in range(start_index, total, BATCH_SIZE):

        batch = semantic_chunks[batch_start : batch_start + BATCH_SIZE]

        ids        = []
        texts      = []
        embeddings = []
        metadatas  = []

        # Collect texts and IDs
        for i, chunk in enumerate(batch):
            ids.append(f"chunk_{batch_start + i}")
            texts.append(chunk["content"])

        # Encode embeddings — CPU optimized batch size 16
        batch_embeddings = model.encode(
            texts,
            batch_size=16,              # Optimized for Intel CPU (reduced from 64)
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        # Build metadata
        for i, chunk in enumerate(batch):
            meta = chunk["metadata"]

            keywords_raw = meta.get("keywords", [])
            if isinstance(keywords_raw, list):
                keywords_str = ", ".join(keywords_raw)
            else:
                keywords_str = str(keywords_raw)

            metadatas.append({
                "source"   : str(meta.get("source", "")),
                "page"     : int(meta.get("page", 0)),
                "category" : str(meta.get("category", "General")),
                "section"  : str(chunk.get("section", "UNKNOWN")),
                "keywords" : keywords_str,
            })
            embeddings.append(batch_embeddings[i].tolist())

        # Upsert to ChromaDB
        collection.upsert(
            ids        = ids,
            documents  = texts,
            embeddings = embeddings,
            metadatas  = metadatas
        )

        end = min(batch_start + BATCH_SIZE, total)
        print(f"✅ Upserted {end}/{total} chunks")


# =========================================
# FINAL SUMMARY
# =========================================

print(f"\n{'='*50}")
print("STORAGE COMPLETE")
print(f"{'='*50}\n")

count = collection.count()
print(f"Total documents in ChromaDB : {count}")
print(f"Collection name             : {COLLECTION_NAME}")
print(f"Stored at                   : {CHROMA_PATH}")
print(f"Device Used                 : {DEVICE.upper()}")
print(f"IPEX Optimized              : {'Yes ✅' if IPEX_AVAILABLE else 'No'}\n")


# =========================================
# SAMPLE ENTRY
# =========================================

print("--- Sample Entry ---")
peek = collection.peek(limit=1)
print(f"ID       : {peek['ids'][0]}")
print(f"Section  : {peek['metadatas'][0]['section']}")
print(f"Category : {peek['metadatas'][0]['category']}")
print(f"Source   : {peek['metadatas'][0]['source']}")
print(f"Page     : {peek['metadatas'][0]['page']}")
print(f"Keywords : {peek['metadatas'][0]['keywords']}")
print(f"Text     : {peek['documents'][0][:300]}\n")


# =========================================
# CATEGORY DISTRIBUTION
# =========================================

print("--- Category Distribution ---\n")

all_meta = collection.get(include=["metadatas"])["metadatas"]
category_counts = {}
for m in all_meta:
    cat = m.get("category", "Unknown")
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat:<25} : {count} chunks")

print(f"\n✅ Done! ChromaDB is ready for RAG queries.\n")