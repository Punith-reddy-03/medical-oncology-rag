import re
import json
import pickle
import os
import torch
import yake
from crewai import Agent, Task, Crew
from langchain_ollama import OllamaLLM
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from load import documents

print("\nStarting Advanced Agentic Chunking (Dynamic Category Mode)...\n")


# =========================================
# CONFIGURATION
# =========================================

BATCH_SIZE = 50               # Reduced from 100 → 50 for CPU
CHECKPOINT_FILE = "chunks_checkpoint.pkl"
FINAL_OUTPUT_FILE = "semantic_chunks.pkl"
CATEGORIES_FILE = "discovered_categories.json"
DISCOVERY_SAMPLE_SIZE = 30    # Reduced from 60 → 30 for CPU


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
# Boosts performance on Intel CPUs & Xe Graphics
# =========================================

IPEX_AVAILABLE = False
if DEVICE == "cpu":
    try:
        import intel_extension_for_pytorch as ipex
        IPEX_AVAILABLE = True
        print("✅ Intel Extension for PyTorch (IPEX) loaded — Intel CPU optimized!\n")
    except ImportError:
        print("ℹ️  IPEX not installed. Running standard CPU mode.")
        print("   To install: pip install intel-extension-for-pytorch\n")

# =========================================
# LOAD SENTENCE TRANSFORMER MODEL
# =========================================

print("Loading SentenceTransformer model...\n")

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)

# Apply IPEX optimization if available
if IPEX_AVAILABLE:
    try:
        # Optimize the underlying PyTorch model with IPEX
        model[0].auto_model = ipex.optimize(model[0].auto_model)
        print("✅ SentenceTransformer model optimized with IPEX!\n")
    except Exception as e:
        print(f"ℹ️  IPEX model optimization skipped: {e}\n")

# Set number of threads for best CPU performance
torch.set_num_threads(os.cpu_count())
print(f"✅ Using {os.cpu_count()} CPU threads for parallel processing\n")


# =========================================
# KEYWORD EXTRACTION MODEL
# =========================================

keyword_extractor = yake.KeywordExtractor(
    lan="en",
    n=2,
    top=10
)


# =========================================
# LLM - MEDGEMMA (via Ollama)
# =========================================

llm = OllamaLLM(model="medgemma")


# =========================================
# CREWAI AGENTS
# =========================================

# --- Agent 1: Category Discovery Agent ---
category_discovery_agent = Agent(
    role="Medical Ontology Discovery Agent",

    goal="""
    Analyze a large collection of medical text samples and
    discover the best set of topic categories that covers
    ALL content found in the books.

    Return a flat list of 8-15 category names that together
    cover every type of medical information present.
    """,

    backstory="""
    You are an expert Medical Knowledge Architect.
    You have read thousands of medical textbooks and PDFs.

    Given a batch of raw text samples from medical PDFs,
    you identify the recurring TYPES of information present
    (e.g. Symptoms, Diagnosis, Drug Mechanisms, Nursing Care,
    Pathophysiology, etc.) and define a clean ontology
    that fits THIS specific book collection.

    You always return structured JSON only.
    No explanations. No preamble. JSON only.
    """,

    llm="ollama/medgemma",
    verbose=False
)


# --- Agent 2: Chunk Classification Agent ---
chunk_classifier_agent = Agent(
    role="Medical Chunk Classification Agent",

    goal="""
    Given a medical text chunk and a list of valid categories,
    classify the chunk into the most appropriate category.
    Also extract the top 5 most important medical keywords.
    """,

    backstory="""
    You are an expert Medical Text Classifier.
    You read a medical text chunk and:
    - Understand the medical content deeply
    - Assign the single best matching category from the provided list
    - Extract the 5 most important medical keywords

    You always return structured JSON only.
    No explanations. No preamble. JSON only.
    """,

    llm="ollama/medgemma",
    verbose=False
)


# =========================================
# PHASE 1: DYNAMIC CATEGORY DISCOVERY
# =========================================

def discover_categories_from_books(documents, sample_size=DISCOVERY_SAMPLE_SIZE):
    """
    Sample text from across all PDFs and let the LLM
    define the best category ontology for THIS book collection.
    """

    print("=" * 50)
    print("PHASE 1: Discovering Categories from Your PDFs")
    print("=" * 50)

    total_docs = len(documents)
    step = max(1, total_docs // sample_size)

    sampled_texts = []
    for i in range(0, total_docs, step):
        doc = documents[i]
        text = doc["content"][:600].strip()
        if len(text.split()) > 30:
            sampled_texts.append(
                f"[Source: {doc['source']} | Page: {doc['page']}]\n{text}"
            )
        if len(sampled_texts) >= sample_size:
            break

    print(f"Sampled {len(sampled_texts)} text chunks from {total_docs} documents.\n")

    combined_sample = "\n\n---\n\n".join(sampled_texts)

    task = Task(
        description=f"""
You are analyzing medical PDFs from a collection of {total_docs} pages.

Below are {len(sampled_texts)} sample text excerpts from across all the books:

{combined_sample[:12000]}

Your task:
1. Read through ALL the samples carefully.
2. Identify the recurring TYPES of medical information present.
3. Define a set of 8 to 15 category names that together cover
   EVERY type of content found in this specific book collection.
4. Category names should be short (1-3 words), clear, and distinct.

Examples of good category names:
  Symptoms, Diagnosis, Treatment, Pathophysiology,
  Drug Therapy, Nursing Care, Anatomy, Staging,
  Prevention, Prognosis, Epidemiology, Patient Education,
  Surgical Procedures, Lab Investigations, Palliative Care

But YOUR categories should fit what is ACTUALLY in these books.
Do not use generic categories if the books don't contain that content.

RETURN ONLY valid JSON:
{{
  "categories": ["Category1", "Category2", "Category3", ...]
}}

No explanations. No markdown. JSON only.
""",
        expected_output='{"categories": [...]}',
        agent=category_discovery_agent
    )

    crew = Crew(
        agents=[category_discovery_agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    try:
        raw = str(result)
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        categories = parsed.get("categories", [])

        if "General" not in categories:
            categories.append("General")

        print(f"✅ Discovered {len(categories)} categories:\n")
        for cat in categories:
            print(f"   → {cat}")
        print()

        return categories

    except Exception as e:
        print(f"⚠️ Category discovery parse failed: {e}")
        print("   Falling back to default medical categories.\n")
        return [
            "Symptoms", "Diagnosis", "Treatment", "Pathophysiology",
            "Drug Therapy", "Nursing Care", "Anatomy", "Staging",
            "Prevention", "Prognosis", "Epidemiology", "General"
        ]


# =========================================
# FAST KEYWORD-BASED PRE-CLASSIFIER
# =========================================

def build_keyword_index(categories):
    index = {}
    for cat in categories:
        words = cat.lower().replace("_", " ").split()
        index[cat] = words
    return index


def map_to_category_fast(heading, chunk_text, categories, keyword_index):
    heading_lower = heading.lower()
    content_lower = chunk_text.lower()

    for cat, words in keyword_index.items():
        for w in words:
            if w in heading_lower:
                return cat

    scores = {}
    for cat, words in keyword_index.items():
        score = sum(1 for w in words if w in content_lower)
        if score > 0:
            scores[cat] = score

    if scores:
        return max(scores, key=scores.get)

    return None


# =========================================
# LLM CLASSIFICATION (FALLBACK)
# =========================================

def classify_with_llm(chunk_text, heading, valid_categories):
    categories_str = ", ".join(valid_categories)

    task = Task(
        description=f"""
Analyze this medical text chunk and classify it.

SECTION HEADING: {heading}

MEDICAL TEXT CHUNK:
{chunk_text[:800]}

VALID CATEGORIES (choose ONLY from this list):
{categories_str}

Your job:
1. Identify the single best category from the list above.
2. Extract the top 5 most important medical keywords from the chunk.

RETURN ONLY valid JSON:
{{
  "category": "one category from the list above",
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]
}}

No explanations. No markdown. JSON only.
""",
        expected_output='{"category": "...", "keywords": [...]}',
        agent=chunk_classifier_agent
    )

    crew = Crew(
        agents=[chunk_classifier_agent],
        tasks=[task],
        verbose=False
    )

    result = crew.kickoff()

    try:
        raw = str(result)
        clean = raw.replace("```json", "").replace("```", "").strip()
        parsed = json.loads(clean)
        category = parsed.get("category", "General")
        keywords = parsed.get("keywords", [])

        if category not in valid_categories:
            for vc in valid_categories:
                if vc.lower() in category.lower() or category.lower() in vc.lower():
                    category = vc
                    break
            else:
                category = "General"

        return category, keywords

    except Exception as e:
        print(f"    [LLM Agent] Parse failed: {e}. Using General.")
        return "General", []


# =========================================
# LOAD OR DISCOVER CATEGORIES
# =========================================

if os.path.exists(CATEGORIES_FILE):
    print(f"Loading previously discovered categories from {CATEGORIES_FILE}...\n")
    with open(CATEGORIES_FILE, "r") as f:
        VALID_CATEGORIES = json.load(f)
    print(f"✅ Loaded {len(VALID_CATEGORIES)} categories: {VALID_CATEGORIES}\n")
else:
    VALID_CATEGORIES = discover_categories_from_books(documents)
    with open(CATEGORIES_FILE, "w") as f:
        json.dump(VALID_CATEGORIES, f, indent=2)
    print(f"✅ Categories saved to {CATEGORIES_FILE}\n")

KEYWORD_INDEX = build_keyword_index(VALID_CATEGORIES)


# =========================================
# LOAD CHECKPOINT IF EXISTS
# =========================================

if os.path.exists(CHECKPOINT_FILE):
    print(f"Checkpoint found! Loading from {CHECKPOINT_FILE}...\n")
    with open(CHECKPOINT_FILE, "rb") as f:
        checkpoint = pickle.load(f)

    semantic_chunks = checkpoint["chunks"]
    start_doc_index = checkpoint["doc_index"]
    llm_classified_count = checkpoint["llm_count"]
    fast_classified_count = checkpoint["fast_count"]

    print(f"Resuming from Document Index: {start_doc_index}")
    print(f"Already Processed Chunks: {len(semantic_chunks)}\n")

else:
    print("No checkpoint found. Starting fresh...\n")
    semantic_chunks = []
    start_doc_index = 0
    llm_classified_count = 0
    fast_classified_count = 0


# =========================================
# HEADING DETECTION PATTERN
# =========================================

heading_pattern = r"\n([A-Z][A-Z\s]{3,})\n"


# =========================================
# PHASE 2: PROCESS DOCUMENTS IN BATCHES
# =========================================

print("=" * 50)
print("PHASE 2: Semantic Chunking + Classification")
print("=" * 50 + "\n")

pending_docs = documents[start_doc_index:]
total_docs = len(documents)
batch = []

for doc_offset, doc in enumerate(pending_docs):

    text = doc["content"]
    source = doc["source"]
    page = doc["page"]
    actual_index = start_doc_index + doc_offset

    print(f"Processing [{actual_index + 1}/{total_docs}]: {source} | Page: {page}")

    sections = re.split(heading_pattern, text)
    current_heading = "UNKNOWN"

    for section in sections:

        cleaned_section = section.strip()
        if not cleaned_section:
            continue

        if (
            cleaned_section.isupper()
            and len(cleaned_section.split()) <= 8
            and len(cleaned_section) < 80
        ):
            current_heading = cleaned_section

        else:
            paragraphs = [
                p.strip()
                for p in cleaned_section.split("\n\n")
                if len(p.strip().split()) >= 30
            ]

            if not paragraphs:
                continue

            # -----------------------------------------------
            # CPU OPTIMIZED: Reduced batch_size from 64 → 16
            # -----------------------------------------------
            embeddings = model.encode(
                paragraphs,
                batch_size=16,              # Optimized for Intel CPU
                show_progress_bar=False,
                convert_to_numpy=True       # Faster on CPU than tensor
            )

            current_chunk = []
            previous_embedding = None

            for para, embedding in zip(paragraphs, embeddings):

                if previous_embedding is None:
                    current_chunk.append(para)
                    previous_embedding = embedding
                    continue

                similarity = cosine_similarity(
                    [previous_embedding], [embedding]
                )[0][0]

                if similarity > 0.7:
                    current_chunk.append(para)
                else:
                    if current_chunk:
                        chunk_text = "\n\n".join(current_chunk)

                        extracted = keyword_extractor.extract_keywords(chunk_text)
                        yake_keywords = [kw[0] for kw in extracted] or ["medical"]

                        category = map_to_category_fast(
                            current_heading, chunk_text,
                            VALID_CATEGORIES, KEYWORD_INDEX
                        )

                        if category is None:
                            category, llm_kws = classify_with_llm(
                                chunk_text, current_heading, VALID_CATEGORIES
                            )
                            llm_classified_count += 1
                            if llm_kws:
                                yake_keywords = list(
                                    dict.fromkeys(llm_kws + yake_keywords)
                                )[:10]
                        else:
                            fast_classified_count += 1

                        batch.append({
                            "section": current_heading,
                            "content": chunk_text,
                            "metadata": {
                                "source": source,
                                "page": page,
                                "keywords": yake_keywords,
                                "category": category
                            }
                        })

                    current_chunk = [para]

                previous_embedding = embedding

            # Store remaining chunk
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)

                extracted = keyword_extractor.extract_keywords(chunk_text)
                yake_keywords = [kw[0] for kw in extracted] or ["medical"]

                category = map_to_category_fast(
                    current_heading, chunk_text,
                    VALID_CATEGORIES, KEYWORD_INDEX
                )

                if category is None:
                    category, llm_kws = classify_with_llm(
                        chunk_text, current_heading, VALID_CATEGORIES
                    )
                    llm_classified_count += 1
                    if llm_kws:
                        yake_keywords = list(
                            dict.fromkeys(llm_kws + yake_keywords)
                        )[:10]
                else:
                    fast_classified_count += 1

                batch.append({
                    "section": current_heading,
                    "content": chunk_text,
                    "metadata": {
                        "source": source,
                        "page": page,
                        "keywords": yake_keywords,
                        "category": category
                    }
                })

    # =========================================
    # SAVE BATCH EVERY 50 DOCUMENTS (reduced)
    # =========================================

    if (doc_offset + 1) % BATCH_SIZE == 0:

        semantic_chunks.extend(batch)
        batch = []

        checkpoint = {
            "chunks": semantic_chunks,
            "doc_index": actual_index + 1,
            "llm_count": llm_classified_count,
            "fast_count": fast_classified_count
        }

        with open(CHECKPOINT_FILE, "wb") as f:
            pickle.dump(checkpoint, f)

        print(f"\n✅ Batch saved at document {actual_index + 1}")
        print(f"   Total chunks so far: {len(semantic_chunks)}\n")


# =========================================
# SAVE FINAL REMAINING BATCH
# =========================================

if batch:
    semantic_chunks.extend(batch)


# =========================================
# SAVE FINAL semantic_chunks.pkl
# =========================================

print("\nSaving final semantic_chunks.pkl...\n")

with open(FINAL_OUTPUT_FILE, "wb") as f:
    pickle.dump(semantic_chunks, f)

print("✅ semantic_chunks.pkl saved successfully!\n")


# =========================================
# CLEAN UP CHECKPOINT
# =========================================

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    print("✅ Checkpoint file removed.\n")


# =========================================
# SAMPLE OUTPUT
# =========================================

for i in range(min(3, len(semantic_chunks))):
    print("\n===================================")
    print(f"SECTION  : {semantic_chunks[i]['section']}")
    print(f"CATEGORY : {semantic_chunks[i]['metadata']['category']}")
    print(f"SOURCE   : {semantic_chunks[i]['metadata']['source']}")
    print(f"PAGE     : {semantic_chunks[i]['metadata']['page']}")
    print(f"KEYWORDS : {semantic_chunks[i]['metadata']['keywords']}")
    print("===================================\n")
    print(semantic_chunks[i]["content"][:500])


# =========================================
# ANALYTICS
# =========================================

print("\n===================================")
print("CHUNKING ANALYTICS")
print("===================================\n")

total_words = sum(len(c["content"].split()) for c in semantic_chunks)

print(f"Total Chunks          : {len(semantic_chunks)}")
print(f"Total Words           : {total_words}")
print(f"Avg Words Per Chunk   : {total_words / len(semantic_chunks):.2f}")
print(f"Total PDFs Processed  : {len(set(c['metadata']['source'] for c in semantic_chunks))}")
print(f"Fast Classified       : {fast_classified_count} chunks")
print(f"LLM Agent Classified  : {llm_classified_count} chunks")
print(f"Device Used           : {DEVICE.upper()}")
print(f"IPEX Optimized        : {'Yes ✅' if IPEX_AVAILABLE else 'No (install with: pip install intel-extension-for-pytorch)'}")


# =========================================
# CATEGORY DISTRIBUTION
# =========================================

print("\n===================================")
print("ONTOLOGY CATEGORY DISTRIBUTION")
print("===================================\n")

category_counts = {}
for chunk in semantic_chunks:
    cat = chunk["metadata"]["category"]
    category_counts[cat] = category_counts.get(cat, 0) + 1

for cat, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {cat:<25} : {count} chunks")


# =========================================
# SHOW DISCOVERED CATEGORIES
# =========================================

print("\n===================================")
print("DISCOVERED CATEGORIES (from YOUR PDFs)")
print("===================================\n")

for cat in VALID_CATEGORIES:
    print(f"  → {cat}")