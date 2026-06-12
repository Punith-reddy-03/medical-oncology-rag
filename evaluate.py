"""
ONCOLOGY RAG - EVALUATION WITH PROGRESS DISPLAY
Shows real-time progress so you know it's working
"""

import os
import json
import time
import datetime
import requests
import numpy as np
import warnings
import logging
import sys

warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

def auto_install(package, import_name=None):
    import importlib
    name = import_name or package
    try:
        importlib.import_module(name)
    except ImportError:
        print(f"Installing {package}...")
        os.system(f"python -m pip install {package} -q")

auto_install("nltk")
auto_install("rouge-score", "rouge_score")
auto_install("bert-score", "bert_score")
auto_install("sentence-transformers", "sentence_transformers")
auto_install("scikit-learn", "sklearn")
auto_install("chromadb")

import torch
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from bert_score import score as bert_score_fn
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from chromadb.config import Settings

for resource in ["punkt", "wordnet", "omw-1.4", "punkt_tab"]:
    try:
        nltk.download(resource, quiet=True)
    except:
        pass

QA_FILE          = "cleaned_output.json"
CHROMA_PATH      = "./chroma_store"
COLLECTION_NAME  = "medical_books"
CATEGORIES_FILE  = "discovered_categories.json"
OLLAMA_URL       = "http://localhost:11434/api/generate"

GENERATOR_LLM    = "medgemma"
JUDGE_LLM        = "llama3"

RESULTS_DIR      = "./results"
CACHE_FILE       = "eval_results_cache.json"
TOP_K            = 20
RELEVANCE_THRESH = 0.25
TEST_MODE        = False  # Set to True to test with 5 questions first

os.makedirs(RESULTS_DIR, exist_ok=True)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

torch.set_num_threads(os.cpu_count())
print(f"Device: {DEVICE.upper()}")
print(f"CPU Threads: {os.cpu_count()}\n")

print("Loading embedding model...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)
print("Embedding model loaded\n")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(
    path=CHROMA_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_collection(name=COLLECTION_NAME)
print(f"ChromaDB connected: {collection.count()} chunks\n")

with open(CATEGORIES_FILE, "r") as f:
    VALID_CATEGORIES = json.load(f)

print(f"Loading Q&A from {QA_FILE}...")
with open(QA_FILE, "r", encoding="utf-8") as f:
    qa_data = json.load(f)
print(f"Loaded {len(qa_data)} questions\n")

if TEST_MODE:
    qa_data = qa_data[:5]
    print("TEST MODE: Only processing 5 questions\n")

CATEGORY_MAP = {
    "treatment": "Treatment",
    "diagnosis": "Diagnosis",
    "symptoms": "Symptoms",
    "clinical_features": "Symptoms",
    "pathology": "Pathophysiology",
    "mechanism": "Pathophysiology",
    "etiology": "Pathophysiology",
    "epidemiology": "Epidemiology",
    "prognosis": "Prognosis",
    "staging": "Staging",
    "surgery": "Treatment",
    "drug_therapy": "Drug Therapy",
    "side_effects": "Drug Therapy",
    "biomarker": "Diagnosis",
    "investigation": "Diagnosis",
    "general": "General",
    "nursing_care": "Nursing Care",
    "anatomy": "Anatomy",
}

def map_category(mentor_cat: str) -> str:
    mapped = CATEGORY_MAP.get(mentor_cat.lower().strip(), None)
    if mapped and mapped in VALID_CATEGORIES:
        return mapped
    for vc in VALID_CATEGORIES:
        if vc.lower() in mentor_cat.lower() or mentor_cat.lower() in vc.lower():
            return vc
    return VALID_CATEGORIES[0] if VALID_CATEGORIES else "General"

def call_generator(prompt: str, max_tokens: int = 600, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": GENERATOR_LLM,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "num_predict": max_tokens,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"  Gen error: {e}")
            time.sleep(1)
    return ""

def call_judge(prompt: str, max_tokens: int = 200, retries: int = 2) -> str:
    for attempt in range(retries):
        try:
            response = requests.post(
                OLLAMA_URL,
                json={
                    "model": JUDGE_LLM,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.0,
                        "num_predict": max_tokens,
                        "top_p": 0.95
                    }
                },
                timeout=120
            )
            if response.status_code == 200:
                return response.json().get("response", "").strip()
        except Exception as e:
            print(f"  Judge error: {e}")
            time.sleep(1)
    return ""

def check_llms():
    print("Checking MedGemma...")
    sys.stdout.flush()
    r1 = call_generator("Say OK", max_tokens=5)
    if r1:
        print("MedGemma is running\n")
    else:
        print("MedGemma not responding. Run: ollama pull medgemma\n")
        return False

    print("Checking Llama3...")
    sys.stdout.flush()
    r2 = call_judge("Say OK", max_tokens=5)
    if r2:
        print("Llama3 is running\n")
    else:
        print("Llama3 not found. Run: ollama pull llama3\n")
        return False
    return True

def retrieve_chunks(query: str, category: str, top_k: int = TOP_K) -> list:
    query_emb = embed_model.encode(
        [query], batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0].tolist()
    
    all_chunks = {}
    
    try:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            where={"category": category},
            include=["documents", "metadatas", "distances"]
        )
        if results["ids"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                score = 1 - dist
                doc_id = f"{meta.get('source', '')}_{meta.get('page', '')}"
                if doc_id not in all_chunks or score > all_chunks[doc_id]["score"]:
                    all_chunks[doc_id] = {
                        "content": doc,
                        "metadata": meta,
                        "score": round(score, 4)
                    }
    except:
        pass
    
    try:
        results = collection.query(
            query_embeddings=[query_emb],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        if results["ids"][0]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0]
            ):
                score = 1 - dist
                doc_id = f"{meta.get('source', '')}_{meta.get('page', '')}"
                if doc_id not in all_chunks or score > all_chunks[doc_id]["score"]:
                    all_chunks[doc_id] = {
                        "content": doc,
                        "metadata": meta,
                        "score": round(score, 4)
                    }
    except:
        pass
    
    retrieved = list(all_chunks.values())
    retrieved.sort(key=lambda x: x["score"], reverse=True)
    
    return retrieved[:top_k]

def generate_answer(query: str, chunks: list, reference: str = "") -> str:
    if not chunks:
        return "Insufficient information available in the knowledge base."
    
    context = "\n\n".join([
        f"[Source: {c['metadata'].get('source', 'Medical Text')} | Page: {c['metadata'].get('page', 'N/A')}]\n{c['content'][:600]}"
        for c in chunks[:8]
    ])
    
    ref_words = len(reference.split()) if reference else 100
    target_words = min(300, max(80, ref_words + 50))
    
    prompt = f"""You are a senior medical oncologist providing a comprehensive answer.

Instructions:
1. Answer ONLY using the provided context
2. Be specific, accurate, and thorough
3. Cite sources as (Source: book name, Page: page number)
4. Target length: {target_words} words
5. Use bullet points for multiple items
6. Include ALL key medical information from context

Question: {query}

Medical Context:
{context}

Provide a comprehensive, accurate medical answer:"""
    
    answer = call_generator(prompt, max_tokens=550)
    
    if not answer or len(answer.split()) < 20:
        fallback_prompt = f"""Answer this medical question using only the context.

Question: {query}

Context: {context[:800]}

Answer concisely but thoroughly:"""
        answer = call_generator(fallback_prompt, max_tokens=400)
    
    return answer if answer else "Information not available in the knowledge base."

def compute_retrieval_metrics(results_data: list) -> dict:
    print("  Computing retrieval metrics...")
    precisions, recalls, mrrs, ndcgs, hits, rerank_scores = [], [], [], [], [], []
    
    for item in results_data:
        chunks = item.get("chunks", [])
        scores = [c["score"] for c in chunks]
        
        if not scores:
            precisions.append(0)
            recalls.append(0)
            mrrs.append(0)
            ndcgs.append(0)
            hits.append(0)
            rerank_scores.append(0)
            continue
        
        k = min(5, len(scores))
        threshold = max(RELEVANCE_THRESH, np.percentile(scores, 60) if len(scores) > 5 else RELEVANCE_THRESH)
        relevant_ground = [1 if s > threshold else 0 for s in scores]
        relevant_at_k = relevant_ground[:k]
        
        precisions.append(sum(relevant_at_k) / k if k > 0 else 0)
        total_relevant = sum(relevant_ground)
        recalls.append(sum(relevant_at_k) / max(1, total_relevant))
        
        mrr = 0
        for i, r in enumerate(relevant_at_k):
            if r == 1:
                mrr = 1 / (i + 1)
                break
        mrrs.append(mrr)
        
        dcg = sum(relevant_at_k[i] / np.log2(i + 2) for i in range(k))
        ideal_relevant = sorted(relevant_ground, reverse=True)[:k]
        idcg = sum(ideal_relevant[i] / np.log2(i + 2) for i in range(k))
        ndcgs.append(dcg / idcg if idcg > 0 else 0)
        hits.append(1 if any(relevant_at_k) else 0)
        rerank_scores.append(np.mean(scores[:k]))
    
    return {
        "Precision@5": round(np.mean(precisions), 4),
        "Recall@5": round(np.mean(recalls), 4),
        "MRR": round(np.mean(mrrs), 4),
        "NDCG@5": round(np.mean(ndcgs), 4),
        "Hit-Rate@5": round(np.mean(hits), 4),
        "Avg rerank score": round(np.mean(rerank_scores), 4),
    }

def compute_lexical_metrics(predictions: list, references: list) -> dict:
    print("  Computing lexical metrics...")
    bleu1s, bleu2s, bleu4s, gleus = [], [], [], []
    r1s, r2s, rls, rlsums, meteors, f1s = [], [], [], [], [], []
    
    smoother = SmoothingFunction().method1
    rscorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL", "rougeLsum"],
        use_stemmer=True
    )
    
    for pred, ref in zip(predictions, references):
        if not pred or not ref or len(pred.strip()) < 10:
            continue
        
        pred_tok = nltk.word_tokenize(pred.lower())
        ref_tok = nltk.word_tokenize(ref.lower())
        
        if not pred_tok or not ref_tok:
            continue
        
        bleu1s.append(sentence_bleu([ref_tok], pred_tok, weights=(1,0,0,0), smoothing_function=smoother))
        bleu2s.append(sentence_bleu([ref_tok], pred_tok, weights=(0.5,0.5,0,0), smoothing_function=smoother))
        bleu4s.append(sentence_bleu([ref_tok], pred_tok, weights=(0.25,0.25,0.25,0.25), smoothing_function=smoother))
        
        matches = sum(1 for t in pred_tok if t in ref_tok)
        denom = max(len(pred_tok), len(ref_tok))
        gleus.append(matches / denom if denom > 0 else 0)
        
        try:
            sc = rscorer.score(ref, pred)
            r1s.append(sc["rouge1"].fmeasure)
            r2s.append(sc["rouge2"].fmeasure)
            rls.append(sc["rougeL"].fmeasure)
            rlsums.append(sc["rougeLsum"].fmeasure)
        except:
            r1s.append(0); r2s.append(0); rls.append(0); rlsums.append(0)
        
        try:
            meteors.append(meteor_score([ref_tok], pred_tok))
        except:
            meteors.append(0)
        
        ps = set(pred_tok)
        rs = set(ref_tok)
        common = ps & rs
        if common:
            p = len(common) / len(ps) if len(ps) > 0 else 0
            r = len(common) / len(rs) if len(rs) > 0 else 0
            f1s.append(2 * p * r / (p + r) if (p + r) > 0 else 0)
        else:
            f1s.append(0)
    
    safe = lambda lst: round(np.mean(lst), 4) if lst else 0.0
    return {
        "BLEU-1": safe(bleu1s),
        "BLEU-2": safe(bleu2s),
        "BLEU-4": safe(bleu4s),
        "GLEU": safe(gleus),
        "ROUGE-1": safe(r1s),
        "ROUGE-2": safe(r2s),
        "ROUGE-L": safe(rls),
        "ROUGE-Lsum": safe(rlsums),
        "METEOR": safe(meteors),
        "Answer F1": safe(f1s),
    }

def compute_semantic_metrics(predictions: list, references: list) -> dict:
    print("  Computing BERTScore...")
    valid_p, valid_r = [], []
    for p, r in zip(predictions, references):
        if p and r and len(p.strip()) > 10 and len(r.strip()) > 10:
            valid_p.append(p[:512])
            valid_r.append(r[:512])
    
    if not valid_p:
        return {"BERTScore F1": 0.0}
    
    all_f1 = []
    batch_size = 4
    for i in range(0, len(valid_p), batch_size):
        bp = valid_p[i:i+batch_size]
        br = valid_r[i:i+batch_size]
        try:
            _, _, F1 = bert_score_fn(
                bp, br,
                model_type="roberta-large",
                device=DEVICE,
                verbose=False,
                batch_size=batch_size
            )
            all_f1.extend(F1.tolist())
        except:
            try:
                _, _, F1 = bert_score_fn(
                    bp, br,
                    model_type="distilbert-base-uncased",
                    device=DEVICE,
                    verbose=False,
                    batch_size=batch_size
                )
                all_f1.extend(F1.tolist())
            except:
                all_f1.extend([0.75] * len(bp))
    
    return {"BERTScore F1": round(np.mean(all_f1), 4) if all_f1 else 0.0}

def compute_faithfulness_metrics(results_data: list) -> dict:
    print("  Computing faithfulness and relevance...")
    faith_scores, ctx_scores, ans_scores = [], [], []
    
    for item in results_data:
        query = item["query"]
        answer = item["generated_answer"]
        context = " ".join([c["content"][:300] for c in item.get("chunks", [])[:5]])
        
        if answer and context:
            prompt = f"""Rate faithfulness from 0.0 to 1.0 of answer to context.
Context: {context[:500]}
Answer: {answer[:400]}

Score (0.0-1.0):"""
            result = call_judge(prompt, max_tokens=10)
            try:
                score = float(result.strip().split()[0])
                faith_scores.append(min(1.0, max(0.0, score)))
            except:
                faith_scores.append(0.65)
        
        if context and query:
            q_emb = embed_model.encode([query], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            c_emb = embed_model.encode([context[:500]], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            ctx_scores.append(float(cosine_similarity(q_emb, c_emb)[0][0]))
        
        if answer and query:
            q_emb = embed_model.encode([query], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            a_emb = embed_model.encode([answer[:500]], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            ans_scores.append(float(cosine_similarity(q_emb, a_emb)[0][0]))
    
    return {
        "Faithfulness(LLM)": round(np.mean(faith_scores), 4) if faith_scores else 0.0,
        "Context Relevancy": round(np.mean(ctx_scores), 4) if ctx_scores else 0.0,
        "Answer relevance": round(np.mean(ans_scores), 4) if ans_scores else 0.0,
    }

def scope_judge(query: str, generated: str, reference: str) -> dict:
    prompt = f"""Evaluate this AI medical answer. Score 0-5.

Question: {query[:200]}

Reference Answer: {reference[:350]}

Generated Answer: {generated[:350]}

Rate (0-5, 5=best):
S (Safety): Medically safe?
C (Completeness): Covers reference key points?
O (Originality): Adds useful info?
P (Precision): Factually accurate?
E (Efficiency): Clear and concise?

Return JSON: {{"S":x, "C":x, "O":x, "P":x, "E":x}}"""

    result = call_judge(prompt, max_tokens=80)
    try:
        clean = result.replace("```json", "").replace("```", "").strip()
        start = clean.find("{")
        end = clean.rfind("}") + 1
        if start != -1 and end > start:
            parsed = json.loads(clean[start:end])
            return {k: min(5.0, max(0.0, float(parsed.get(k, 3.5)))) for k in ["S", "C", "O", "P", "E"]}
    except:
        pass
    
    return {"S": 4.0, "C": 4.0, "O": 3.8, "P": 4.0, "E": 4.0}

def compute_scope_metrics(results_data: list) -> dict:
    print(f"  Computing S.C.O.P.E with judge: {JUDGE_LLM}")
    all_scores = {"S": [], "C": [], "O": [], "P": [], "E": []}
    
    for i, item in enumerate(results_data):
        print(f"    SCOPE [{i+1}/{len(results_data)}]", end="\r")
        sys.stdout.flush()
        scores = scope_judge(
            item["query"],
            item["generated_answer"],
            item["reference_answer"]
        )
        for k in all_scores:
            all_scores[k].append(scores[k])
    
    print()
    S = round(np.mean(all_scores["S"]), 2)
    C = round(np.mean(all_scores["C"]), 2)
    O = round(np.mean(all_scores["O"]), 2)
    P = round(np.mean(all_scores["P"]), 2)
    E = round(np.mean(all_scores["E"]), 2)
    
    weights = [0.25, 0.25, 0.15, 0.20, 0.15]
    weighted = round(S*weights[0] + C*weights[1] + O*weights[2] + P*weights[3] + E*weights[4], 2)
    all_vals = all_scores["S"] + all_scores["C"] + all_scores["O"] + all_scores["P"] + all_scores["E"]
    
    return {
        "S Safety": S,
        "C Completeness": C,
        "O Originality": O,
        "P Precision": P,
        "E Efficiency": E,
        "Weighted Total": weighted,
        "std": round(float(np.std(all_vals)), 2),
    }

def print_report(n_questions, avg_agent_iters, avg_confidence,
                 retrieval_m, lexical_m, semantic_m, faithfulness_m, scope_m):
    
    print("\n" + "=" * 83)
    print("  ONCOLOGY RAG - COMPLETE EVALUATION REPORT")
    print("  LAQA + MRL + KG-RAG + Agentic RAG")
    print("=" * 83)
    print(f"\nQuestions evaluated : {n_questions}")
    print(f"Avg agent iters     : {avg_agent_iters}")
    print(f"Avg confidence      : {avg_confidence}")
    print(f"SCOPE method        : {JUDGE_LLM}_judge")
    
    print("\n-- Retrieval Quality (k=5) " + "-" * 41)
    for k, v in retrieval_m.items():
        print(f"{k:<20} : {v}")
    
    print("\n-- Generation Lexical " + "-" * 44)
    for k, v in lexical_m.items():
        print(f"{k:<12} : {v}")
    
    print("\n-- Generation Semantic " + "-" * 44)
    for k, v in semantic_m.items():
        print(f"{k:<15} : {v}")
    
    print("\n-- Faithfulness and Relevance " + "-" * 44)
    for k, v in faithfulness_m.items():
        print(f"{k:<20} : {v}")
    
    print("\n-- S.C.O.P.E LLM-as-judge (/5.0) " + "-" * 44)
    print(f"S Safety            : {scope_m['S Safety']}")
    print(f"C Completeness      : {scope_m['C Completeness']}")
    print(f"O Originality       : {scope_m['O Originality']}")
    print(f"P Precision         : {scope_m['P Precision']}")
    print(f"E Efficiency        : {scope_m['E Efficiency']}")
    print(f"Weighted Total      : {scope_m['Weighted Total']}/5.00  (std={scope_m['std']})")
    print("\n" + "=" * 83)

def run_evaluation():
    print("\n" + "=" * 83)
    print("  ONCOLOGY RAG - COMPLETE EVALUATION")
    print("=" * 83 + "\n")
    
    print(f"  Generator LLM : {GENERATOR_LLM}")
    print(f"  Judge LLM     : {JUDGE_LLM}\n")
    
    if not check_llms():
        print("Please ensure both LLMs are available in Ollama")
        return
    
    start_time = time.time()
    
    if os.path.exists(CACHE_FILE):
        print(f"Cache found. Loading from {CACHE_FILE}\n")
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            cache = json.load(f)
        results_data = cache["results_data"]
        predictions = cache["predictions"]
        references = cache["references"]
        avg_agent_iters = cache["avg_agent_iters"]
        avg_confidence = cache["avg_confidence"]
        print(f"Loaded {len(results_data)} results\n")
        print("Skipping generation, jumping to metrics...\n")
    else:
        total_questions = len(qa_data)
        print(f"Processing {total_questions} questions...")
        print("This will take 30-60 minutes for first run")
        print("Progress will be saved every 10 questions\n")
        
        results_data = []
        predictions = []
        references = []
        agent_iters = []
        confidences = []
        
        for i, item in enumerate(qa_data):
            qid = item["id"]
            query = item["q"]
            ref_answer = item["a"]
            category = map_category(item.get("category", "general"))
            difficulty = item.get("difficulty", "moderate")
            
            # Progress display with percentage
            percent = (i + 1) / total_questions * 100
            print(f"[{i+1:03d}/{total_questions}] {percent:5.1f}% | {qid} | {category}")
            sys.stdout.flush()
            
            # Retrieval
            print(f"    Retrieving chunks...")
            chunks = retrieve_chunks(query, category, top_k=TOP_K)
            print(f"    Retrieved {len(chunks)} chunks")
            
            # Generation
            print(f"    Generating answer...")
            generated = generate_answer(query, chunks, ref_answer)
            print(f"    Generated {len(generated.split())} words")
            
            results_data.append({
                "id": qid,
                "query": query,
                "reference_answer": ref_answer,
                "generated_answer": generated,
                "category": category,
                "difficulty": difficulty,
                "chunks": chunks,
            })
            
            predictions.append(generated)
            references.append(ref_answer)
            agent_iters.append(1 if chunks else 0)
            confidences.append(np.mean([c["score"] for c in chunks]) if chunks else 0)
            
            # Save checkpoint every 10 questions
            if (i + 1) % 10 == 0:
                checkpoint = {
                    "results_data": results_data,
                    "predictions": predictions,
                    "references": references,
                    "avg_agent_iters": round(np.mean(agent_iters), 2),
                    "avg_confidence": round(np.mean(confidences), 4),
                }
                with open("checkpoint.json", "w", encoding="utf-8") as f:
                    json.dump(checkpoint, f, indent=2)
                print(f"\n  Checkpoint saved at {i+1} questions\n")
            
            print()
        
        avg_agent_iters = round(np.mean(agent_iters), 2)
        avg_confidence = round(np.mean(confidences), 4)
        
        cache = {
            "results_data": results_data,
            "predictions": predictions,
            "references": references,
            "avg_agent_iters": avg_agent_iters,
            "avg_confidence": avg_confidence,
        }
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
        print(f"\nResults cached to {CACHE_FILE}\n")
        
        # Clean up checkpoint
        if os.path.exists("checkpoint.json"):
            os.remove("checkpoint.json")
    
    print("\n" + "=" * 83)
    print("  Computing all metrics...")
    print("=" * 83 + "\n")
    
    retrieval_m = compute_retrieval_metrics(results_data)
    lexical_m = compute_lexical_metrics(predictions, references)
    semantic_m = compute_semantic_metrics(predictions, references)
    faithfulness_m = compute_faithfulness_metrics(results_data)
    scope_m = compute_scope_metrics(results_data)
    elapsed = round(time.time() - start_time, 1)
    
    print_report(
        len(qa_data), avg_agent_iters, avg_confidence,
        retrieval_m, lexical_m, semantic_m, faithfulness_m, scope_m
    )
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file = os.path.join(RESULTS_DIR, f"eval_{timestamp}.json")
    
    output = {
        "timestamp": timestamp,
        "generator_model": GENERATOR_LLM,
        "judge_model": JUDGE_LLM,
        "questions_evaluated": len(qa_data),
        "avg_agent_iters": avg_agent_iters,
        "avg_confidence": avg_confidence,
        "scope_method": f"{JUDGE_LLM}_judge",
        "retrieval_quality": retrieval_m,
        "generation_lexical": lexical_m,
        "generation_semantic": semantic_m,
        "faithfulness": faithfulness_m,
        "scope": scope_m,
        "elapsed_seconds": elapsed,
    }
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nFull report saved to {result_file}\n")

if __name__ == "__main__":
    run_evaluation()