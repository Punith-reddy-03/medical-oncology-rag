import os
import json
import requests

# =========================================
# FIX: Force Ollama — No OpenAI needed
# =========================================
os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "ollama/medgemma"

OLLAMA_URL   = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "medgemma"

# =========================================
# LOAD VALID CATEGORIES
# =========================================

with open("discovered_categories.json", "r") as f:
    VALID_CATEGORIES = json.load(f)

CATEGORIES_STR = ", ".join(VALID_CATEGORIES)


# =========================================
# DIRECT OLLAMA CALL (No CrewAI / No OpenAI)
# =========================================

def call_ollama(prompt: str, retries: int = 3) -> str:
    """
    Call Ollama MedGemma directly via HTTP.
    No CrewAI, no OpenAI dependency.
    """
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": OLLAMA_MODEL,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": 500
                    }
                },
                timeout=180
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
            else:
                print(f"    [Ollama] HTTP {response.status_code} on attempt {attempt+1}")
        except requests.exceptions.ConnectionError:
            print(f"    [Ollama] Connection failed on attempt {attempt+1}.")
            print("    Make sure Ollama is running: ollama serve")
        except Exception as e:
            print(f"    [Ollama] Error on attempt {attempt+1}: {e}")
    return None


# =========================================
# CHECK OLLAMA IS RUNNING
# =========================================

def check_ollama():
    print("Checking Ollama connection...")
    test = call_ollama("Say OK")
    if test:
        print(f"✅ Ollama is running!\n")
        return True
    else:
        print("❌ ERROR: Ollama is not running!")
        print("   Open a NEW terminal and run: ollama serve")
        print("   Then run this script again.\n")
        return False


# =========================================
# AGENT 1: QUERY ANALYSER
# Extracts category + keywords from user question
# =========================================

def query_analyser_agent(user_query: str) -> dict:
    """
    Analyse the user medical question and extract:
    1. The single best matching category
    2. Top 5 most important medical keywords
    Returns a dict: {"category": "...", "keywords": [...]}
    """

    prompt = f"""You are an expert medical query understanding system.

Analyse this medical question from the user:
"{user_query}"

Your job:
1. Pick the single best category from this list: {CATEGORIES_STR}
2. Extract the top 5 most important medical keywords from the question.

Return ONLY valid JSON in this exact format:
{{"category": "one category from the list above", "keywords": ["keyword1", "keyword2", "keyword3", "keyword4", "keyword5"]}}

No explanations. No markdown. JSON only."""

    result = call_ollama(prompt)

    if result is None:
        print("⚠️ Query analyser failed. Using defaults.")
        return {"category": "General", "keywords": user_query.split()[:5]}

    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start != -1 and end > start:
            clean = clean[start:end]

        parsed   = json.loads(clean)
        category = parsed.get("category", "General")
        keywords = parsed.get("keywords", [])

        # Validate category
        if category not in VALID_CATEGORIES:
            for vc in VALID_CATEGORIES:
                if vc.lower() in category.lower() or category.lower() in vc.lower():
                    category = vc
                    break
            else:
                category = "General"

        print(f"✅ Query Analysis Done:")
        print(f"   Category : {category}")
        print(f"   Keywords : {keywords}\n")

        return {"category": category, "keywords": keywords}

    except Exception as e:
        print(f"⚠️ Query analyser parse failed: {e}. Using defaults.")
        return {"category": "General", "keywords": user_query.split()[:5]}


# =========================================
# AGENT 2: DOCTOR AGENT
# Answers the user question using retrieved chunks
# =========================================

def doctor_agent(user_query: str, retrieved_chunks: str) -> str:
    """
    Read the retrieved medical text chunks and answer the user question
    in a warm, empathetic, clear doctor style with source citations.
    """

    prompt = f"""You are a senior experienced empathetic medical doctor.

The user asked:
"{user_query}"

Below are the most relevant medical text chunks retrieved from
a collection of real medical textbooks:

{retrieved_chunks}

Your job:
1. Read all the retrieved chunks carefully.
2. Answer the user question in a warm, empathetic, clear doctor style.
3. For every point you make cite the source book and page number like this:
   (Source: <book name>, Page: <page number>)
4. If the chunks do not contain enough information say so honestly.
5. Never make up or hallucinate any medical information.
6. Only use information from the retrieved chunks above.

Structure your answer as:
- A warm opening addressing the patient
- Clear explanation of the answer point by point
- Each point cited with source and page number
- A caring closing note advising them to consult their doctor"""

    result = call_ollama(prompt)

    if result is None:
        return "I'm sorry, I was unable to generate a response at this time. Please ensure Ollama is running and try again."

    return result


# =========================================
# HELPER: FORMAT CHUNKS FOR DOCTOR AGENT
# =========================================

def format_chunks_for_prompt(results: list) -> str:
    """
    Format hybrid search results into a readable string
    for the doctor agent prompt.
    """
    formatted = []
    for r in results:
        chunk_text = f"""--- Chunk {r['rank']} ---
Source   : {r['metadata']['source']}
Page     : {r['metadata']['page']}
Category : {r['metadata']['category']}
Section  : {r['metadata']['section']}
Content  : {r['content'][:600]}
"""
        formatted.append(chunk_text)

    return "\n".join(formatted)


# =========================================
# MAIN PIPELINE: QUERY → RETRIEVE → ANSWER
# =========================================

def run_medical_rag(user_query: str, retriever_fn) -> str:
    """
    Full RAG pipeline:
    1. Analyse query → get category + keywords
    2. Retrieve relevant chunks via hybrid search
    3. Generate doctor-style answer

    Args:
        user_query   : The user's medical question
        retriever_fn : hybrid_search function from retriever.py

    Returns:
        Doctor's answer as a string
    """

    print("=" * 60)
    print("MEDICAL RAG PIPELINE")
    print("=" * 60 + "\n")

    # Step 1: Analyse query
    print("Step 1: Analysing query...\n")
    analysis = query_analyser_agent(user_query)
    category = analysis["category"]
    keywords = analysis["keywords"]

    # Step 2: Retrieve chunks
    print(f"Step 2: Retrieving chunks for category '{category}'...\n")
    results = retriever_fn(user_query, category, top_k=10)

    if not results:
        return f"No relevant medical information found for category '{category}'. Please try a different question."

    print(f"✅ Retrieved {len(results)} chunks\n")

    # Step 3: Format chunks
    retrieved_chunks = format_chunks_for_prompt(results)

    # Step 4: Generate doctor answer
    print("Step 3: Generating doctor response...\n")
    answer = doctor_agent(user_query, retrieved_chunks)

    return answer


# =========================================
# QUICK TEST
# =========================================

if __name__ == "__main__":

    # Check Ollama is running first
    if not check_ollama():
        exit(1)

    # Test query analyser
    print("=" * 60)
    print("TEST: Query Analyser Agent")
    print("=" * 60)

    test_query = "What are the treatment options for lung cancer?"
    result = query_analyser_agent(test_query)
    print(f"Result: {result}\n")

    # Test doctor agent with sample chunks
    print("=" * 60)
    print("TEST: Doctor Agent")
    print("=" * 60)

    sample_chunks = """--- Chunk 1 ---
Source   : cancer-principles-and-practice-of-oncology-6e.pdf
Page     : 512
Category : Treatment
Section  : LUNG CANCER TREATMENT
Content  : Surgery remains the primary treatment for early-stage non-small cell lung cancer.
Lobectomy is the preferred surgical procedure. Chemotherapy with platinum-based regimens
is recommended for advanced stages. Immunotherapy with checkpoint inhibitors has shown
significant benefit in patients with high PD-L1 expression."""

    answer = doctor_agent(test_query, sample_chunks)
    print("\n--- Doctor Response ---\n")
    print(answer)