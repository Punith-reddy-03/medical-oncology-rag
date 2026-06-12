"""
=============================================================
MEDICAL RAG SYSTEM WITH REINFORCEMENT LEARNING
- Learns from every interaction
- Improves answers over time automatically
- Stores feedback and learns from mistakes
- Lightweight RL 
=============================================================
"""

import os
import json
import time
import requests
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

os.environ["OPENAI_API_KEY"] = "NA"
os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
os.environ["OPENAI_MODEL_NAME"] = "ollama/medgemma"

# =========================================
# IMPORTS FROM YOUR PROJECT
# =========================================

from agents import query_analyser_agent, doctor_agent, check_ollama
from retriever import hybrid_search

# =========================================
# CONFIGURATION
# =========================================

CATEGORIES_FILE   = "discovered_categories.json"
OLLAMA_URL        = "http://localhost:11434/api/generate"
OLLAMA_MODEL      = "medgemma"
RL_MEMORY_FILE    = "rl_memory.json"       # stores all past interactions
RL_POLICY_FILE    = "rl_policy.json"       # stores learned improvements
TOP_K             = 10
RL_REWARD_THRESH  = 0.6                    # min score to consider answer good
MAX_MEMORY        = 500                    # max interactions to remember

# =========================================
# DEVICE
# =========================================

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

torch.set_num_threads(os.cpu_count())

# =========================================
# LOAD COMPONENTS
# =========================================

with open(CATEGORIES_FILE, "r") as f:
    VALID_CATEGORIES = json.load(f)

print("Loading embedding model for RL scoring...")
embed_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device=DEVICE
)
print("✅ Embedding model loaded\n")


# =========================================
# OLLAMA CALL
# =========================================

def call_ollama(prompt: str, max_tokens: int = 400) -> str:
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": max_tokens}
            },
            timeout=180
        )
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    except Exception as e:
        print(f"  [Ollama] Error: {e}")
    return ""


# =========================================
# RL MEMORY SYSTEM
# Stores all past Q&A interactions with scores
# =========================================

class RLMemory:
    """
    Lightweight Reinforcement Learning Memory.
    Stores interactions, scores, and learned improvements.
    Works entirely on CPU — no GPU needed.
    """

    def __init__(self):
        self.memory = []    
        self.policy = {}    # learned query→improvement mappings
        self.load()

    def load(self):
        """Load existing memory and policy from disk."""
        if os.path.exists(RL_MEMORY_FILE):
            try:
                with open(RL_MEMORY_FILE, "r", encoding="utf-8") as f:
                    self.memory = json.load(f)
                print(f"✅ RL Memory loaded: {len(self.memory)} past interactions\n")
            except:
                self.memory = []

        if os.path.exists(RL_POLICY_FILE):
            try:
                with open(RL_POLICY_FILE, "r", encoding="utf-8") as f:
                    self.policy = json.load(f)
                print(f"✅ RL Policy loaded: {len(self.policy)} learned improvements\n")
            except:
                self.policy = {}

    def save(self):
        """Save memory and policy to disk."""
        # Keep only last MAX_MEMORY interactions
        if len(self.memory) > MAX_MEMORY:
            self.memory = self.memory[-MAX_MEMORY:]

        with open(RL_MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.memory, f, indent=2, ensure_ascii=False)

        with open(RL_POLICY_FILE, "w", encoding="utf-8") as f:
            json.dump(self.policy, f, indent=2, ensure_ascii=False)

    def add_interaction(self, query: str, answer: str, reward: float,
                        category: str, chunks_used: int):
        """Store a new interaction with its reward score."""
        self.memory.append({
            "timestamp"  : datetime.now().isoformat(),
            "query"      : query,
            "answer"     : answer,
            "reward"     : reward,
            "category"   : category,
            "chunks_used": chunks_used,
        })

    def get_similar_past_interactions(self, query: str, top_k: int = 3) -> list:
        """
        Find past interactions similar to current query.
        Uses cosine similarity on embeddings.
        Returns top_k most similar past Q&A pairs.
        """
        if not self.memory:
            return []

        query_emb = embed_model.encode(
            [query], batch_size=1,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        past_queries = [m["query"] for m in self.memory]

        # Batch encode past queries
        past_embs = embed_model.encode(
            past_queries,
            batch_size=16,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        similarities = cosine_similarity(query_emb, past_embs)[0]
        top_indices  = np.argsort(similarities)[::-1][:top_k]

        similar = []
        for idx in top_indices:
            if similarities[idx] > 0.5:  # only if reasonably similar
                similar.append({
                    "query"     : self.memory[idx]["query"],
                    "answer"    : self.memory[idx]["answer"],
                    "reward"    : self.memory[idx]["reward"],
                    "similarity": round(float(similarities[idx]), 4),
                })
        return similar

    def get_high_reward_examples(self, category: str, top_k: int = 2) -> list:
        """Get best past answers for a given category (reward > threshold)."""
        category_memories = [
            m for m in self.memory
            if m.get("category") == category and m.get("reward", 0) >= RL_REWARD_THRESH
        ]
        # Sort by reward descending
        category_memories.sort(key=lambda x: x["reward"], reverse=True)
        return category_memories[:top_k]

    def get_low_reward_examples(self, category: str, top_k: int = 2) -> list:
        """Get worst past answers to learn what NOT to do."""
        category_memories = [
            m for m in self.memory
            if m.get("category") == category and m.get("reward", 0) < RL_REWARD_THRESH
        ]
        category_memories.sort(key=lambda x: x["reward"])
        return category_memories[:top_k]

    def update_policy(self, query: str, improvement: str, reward: float):
        """Store a learned improvement for a query pattern."""
        key = query[:100]  # use first 100 chars as key
        self.policy[key] = {
            "improvement": improvement,
            "reward"     : reward,
            "timestamp"  : datetime.now().isoformat(),
        }

    def get_stats(self) -> dict:
        """Get RL learning statistics."""
        if not self.memory:
            return {"total": 0, "avg_reward": 0, "high_reward": 0, "low_reward": 0, "improvement": "0.0% good answers"}

        rewards     = [m["reward"] for m in self.memory]
        high_reward = sum(1 for r in rewards if r >= RL_REWARD_THRESH)
        low_reward  = sum(1 for r in rewards if r < RL_REWARD_THRESH)

        return {
            "total"      : len(self.memory),
            "avg_reward" : round(np.mean(rewards), 4),
            "high_reward": high_reward,
            "low_reward" : low_reward,
            "improvement": f"{round(high_reward/len(rewards)*100, 1)}% good answers",
        }


# Initialize RL Memory
rl_memory = RLMemory()


# =========================================
# RL REWARD FUNCTION
# Automatically scores the generated answer
# =========================================

def compute_reward(query: str, answer: str, chunks: list) -> float:
    """
    Compute reward score for a generated answer (0.0 to 1.0).
    Higher reward = better answer.
    Uses multiple signals:
    1. Answer length (too short = bad)
    2. Semantic similarity to query
    3. Semantic similarity to retrieved chunks
    4. LLM self-evaluation score
    """

    if not answer or len(answer.strip()) < 20:
        return 0.1

    scores = []

    # --- Signal 1: Length score ---
    words      = len(answer.split())
    len_score  = min(1.0, words / 100)  # ideal ~100 words
    scores.append(len_score)

    # --- Signal 2: Query relevance ---
    try:
        q_emb = embed_model.encode([query],  batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        a_emb = embed_model.encode([answer[:400]], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
        query_sim = float(cosine_similarity(q_emb, a_emb)[0][0])
        scores.append(max(0, query_sim))
    except:
        scores.append(0.5)

    # --- Signal 3: Context grounding ---
    if chunks:
        try:
            context = " ".join([c["content"][:200] for c in chunks[:3]])
            c_emb   = embed_model.encode([context], batch_size=1, convert_to_numpy=True, normalize_embeddings=True)
            ctx_sim = float(cosine_similarity(a_emb, c_emb)[0][0])
            scores.append(max(0, ctx_sim))
        except:
            scores.append(0.5)

    # --- Signal 4: LLM self-evaluation ---
    eval_prompt = f"""Rate the quality of this medical answer from 0.0 to 1.0.
Question: {query[:150]}
Answer: {answer[:200]}

Consider: accuracy, completeness, clarity.
Reply with ONLY a number between 0.0 and 1.0."""

    eval_result = call_ollama(eval_prompt, max_tokens=10)
    try:
        llm_score = float(eval_result.strip().split()[0])
        llm_score = min(1.0, max(0.0, llm_score))
        scores.append(llm_score)
    except:
        scores.append(0.5)

    # Weighted average
    weights = [0.1, 0.3, 0.3, 0.3]
    reward  = sum(w * s for w, s in zip(weights, scores))
    return round(reward, 4)


# =========================================
# RL-ENHANCED ANSWER GENERATION
# Uses past experience to improve answers
# =========================================

def generate_rl_enhanced_answer(query: str, chunks: list,
                                  category: str, similar_past: list,
                                  high_examples: list, low_examples: list) -> str:
    """
    Generate answer enriched with RL knowledge:
    - Uses high-reward past answers as positive examples
    - Avoids patterns from low-reward past answers
    - Incorporates learned improvements
    """

    if not chunks:
        return "Insufficient information available in the knowledge base."

    context = "\n\n".join([
        f"[Source: {c['metadata'].get('source','?')} | Page: {c['metadata'].get('page','?')}]\n{c['content'][:350]}"
        for c in chunks[:5]
    ])

    # Build RL guidance section
    rl_guidance = ""

    if high_examples:
        rl_guidance += "\n\nEXAMPLES OF HIGH-QUALITY ANSWERS (learn from these):\n"
        for ex in high_examples[:2]:
            rl_guidance += f"Q: {ex['query'][:100]}\nA: {ex['answer'][:200]}\n---\n"

    if low_examples:
        rl_guidance += "\n\nEXAMPLES OF POOR ANSWERS (avoid these patterns):\n"
        for ex in low_examples[:1]:
            rl_guidance += f"Q: {ex['query'][:100]}\nA (BAD): {ex['answer'][:150]}\n---\n"

    if similar_past:
        best_past = max(similar_past, key=lambda x: x["reward"])
        if best_past["reward"] >= RL_REWARD_THRESH:
            rl_guidance += f"\n\nSIMILAR QUESTION WAS ASKED BEFORE (reward={best_past['reward']}):\n"
            rl_guidance += f"Q: {best_past['query'][:100]}\n"
            rl_guidance += f"A: {best_past['answer'][:200]}\n"
            rl_guidance += "Use this as reference but improve upon it.\n"

    prompt = f"""You are a warm, friendly, and caring doctor — like a trusted best friend who is also a doctor.

Your personality:
- Greet the patient warmly by acknowledging their concern
- Speak in simple, everyday friendly language (not cold clinical language)
- Be caring, supportive and conversational — like chatting with a close friend
- Explain medical terms simply when you use them
- Use a few emojis to feel warm 😊❤️
- Cite sources naturally: (from <book name>, page <number>)
- At the end, suggest 2 follow-up questions they might want to ask next
- End with an encouraging, uplifting closing message

{rl_guidance}

Patient asked: {query}

Medical Context from Books:
{context}

Write a warm, friendly, conversational and helpful doctor response:"""

    return call_ollama(prompt, max_tokens=500)


# =========================================
# RL SELF-IMPROVEMENT
# Identifies mistakes and learns from them
# =========================================

def rl_self_improve(query: str, answer: str, reward: float, category: str):
    """
    If reward is low, ask the LLM to identify what went wrong
    and store the improvement for future use.
    """

    if reward >= RL_REWARD_THRESH:
        return  # Answer was good, no improvement needed

    improve_prompt = f"""This medical answer received a low quality score of {reward:.2f}/1.0.

Question: {query[:150]}
Answer: {answer[:300]}

Identify in ONE sentence what was wrong and how to improve it.
Be specific and actionable."""

    improvement = call_ollama(improve_prompt, max_tokens=100)

    if improvement:
        rl_memory.update_policy(query, improvement, reward)
        print(f"  🔄 RL Learning: {improvement[:80]}...")


# =========================================
# FORMAT CHUNKS FOR DISPLAY
# =========================================

def format_chunks_for_doctor(results: list) -> str:
    formatted = ""
    for r in results:
        formatted += f"""
Source   : {r['metadata']['source']}
Page     : {r['metadata']['page']}
Section  : {r['metadata']['section']}
Category : {r['metadata']['category']}
Content  : {r['content'][:300]}
---
"""
    return formatted


# =========================================
# MAIN RUN FUNCTION WITH RL
# =========================================

def run(user_query: str):

    print(f"\nUser Query: {user_query}\n")
    print("=" * 60)

    # --- Step 1: Analyse Query ---
    print("\nStep 1: Analysing query...\n")
    analysis = query_analyser_agent(user_query)
    category = analysis["category"]
    keywords = analysis["keywords"]

    print(f"Extracted Category : {category}")
    print(f"Extracted Keywords : {keywords}\n")

    # --- Step 2: Hybrid Search ---
    print("\nStep 2: Running hybrid search...\n")
    full_query = user_query + " " + " ".join(keywords)
    results    = hybrid_search(query=full_query, category=category, top_k=TOP_K)

    if not results:
        print(f"⚠️ No chunks found for category: '{category}'\n")
        return None

    print(f"✅ Retrieved {len(results)} chunks from ChromaDB\n")

    # --- Step 3: RL Memory Lookup ---
    print("\nStep 3: Checking RL memory for similar past interactions...\n")
    similar_past  = rl_memory.get_similar_past_interactions(user_query, top_k=3)
    high_examples = rl_memory.get_high_reward_examples(category, top_k=2)
    low_examples  = rl_memory.get_low_reward_examples(category, top_k=1)

    if similar_past:
        print(f"  Found {len(similar_past)} similar past interactions")
        best = max(similar_past, key=lambda x: x["reward"])
        print(f"  Best match similarity: {best['similarity']} | reward: {best['reward']}\n")
    else:
        print("  No similar past interactions found (learning from scratch)\n")

    if high_examples:
        print(f"  Found {len(high_examples)} high-quality examples for category '{category}'\n")

    # --- Step 4: Generate RL-Enhanced Answer ---
    print("\nStep 4: Generating RL-enhanced answer...\n")
    final_answer = generate_rl_enhanced_answer(
        user_query, results, category,
        similar_past, high_examples, low_examples
    )

    # --- Step 5: Compute Reward ---
    print("\nStep 5: Computing reward score...\n")
    reward = compute_reward(user_query, final_answer, results)
    print(f"  Reward Score: {reward:.4f} / 1.0  ({'✅ Good' if reward >= RL_REWARD_THRESH else '⚠️ Needs improvement'})\n")

    # --- Step 6: Store in RL Memory ---
    rl_memory.add_interaction(
        query       = user_query,
        answer      = final_answer,
        reward      = reward,
        category    = category,
        chunks_used = len(results)
    )

    # --- Step 7: Self-Improve if reward is low ---
    if reward < RL_REWARD_THRESH:
        print("\nStep 7: RL self-improvement triggered...\n")
        rl_self_improve(user_query, final_answer, reward, category)

    # --- Save updated memory ---
    rl_memory.save()

    print("\n" + "=" * 60)
    print("💬 DOCTOR'S RESPONSE")
    print("=" * 60 + "\n")
    print(final_answer)

    # --- Print RL Stats ---
    stats = rl_memory.get_stats()
    print(f"\n📊 Learning Stats: {stats['total']} interactions | "
      f"Avg quality: {stats['avg_reward']} | "
      f"{stats['improvement']}"
      )
    print("\n💡 Feel free to ask another question — I'm here to help! 😊")
    return final_answer
    



# =========================================
# SHOW RL LEARNING PROGRESS
# =========================================

def show_rl_progress():
    stats = rl_memory.get_stats()
    print("\n" + "=" * 60)
    print("RL LEARNING PROGRESS")
    print("=" * 60)
    print(f"Total interactions : {stats['total']}")
    print(f"Avg reward score   : {stats['avg_reward']}")
    print(f"High quality (≥{RL_REWARD_THRESH}) : {stats['high_reward']}")
    print(f"Low quality  (<{RL_REWARD_THRESH}) : {stats['low_reward']}")
    print(f"Overall quality    : {stats['improvement']}")
    print(f"Learned policies   : {len(rl_memory.policy)}")

    if rl_memory.memory:
        # Show improvement trend
        recent  = rl_memory.memory[-10:] if len(rl_memory.memory) >= 10 else rl_memory.memory
        old     = rl_memory.memory[:10]  if len(rl_memory.memory) >= 10 else []
        r_avg   = round(np.mean([m["reward"] for m in recent]), 4)
        o_avg   = round(np.mean([m["reward"] for m in old]), 4) if old else r_avg
        trend   = "📈 Improving" if r_avg > o_avg else "📉 Needs more interactions"
        print(f"Learning trend     : {trend} (early: {o_avg} → recent: {r_avg})")

    print("=" * 60 + "\n")


# =========================================
# MAIN ENTRY POINT
# =========================================

if __name__ == "__main__":

    print("\n" + "=" * 60)
    print("   👨‍⚕️  YOUR FRIENDLY MEDICAL ASSISTANT  👨‍⚕️")
    print("   Powered by RAG + Reinforcement Learning")
    print("=" * 60 + "\n")

    # Check Ollama
    if not check_ollama():
        print("Please start Ollama first: ollama serve")
        exit(1)

    print("Hi there! 👋 I'm your personal medical assistant.")
    print("Ask me anything about cancer, treatments, symptoms,")
    print("or any medical question — I'm here to help! 😊\n")
    print("Commands:")
    print("  'stats'  → see how much I've learned so far")
    print("  'reset'  → clear my memory")
    print("  'exit'   → goodbye\n")

    while True:
        try:
            query = input("🩺 What's your question for me today? → ").strip()
        except KeyboardInterrupt:
            print("\n\nTake care! Stay healthy! 💚")
            break

        if not query:
            print("Go ahead, don't be shy! Ask me anything 😊\n")
            continue

        if query.lower() in ("exit", "quit", "q"):
            print("\nTake care! Remember — your health is your wealth! 💚 Goodbye!\n")
            break

        if query.lower() == "stats":
            show_rl_progress()
            continue

        if query.lower() == "reset":
            rl_memory.memory = []
            rl_memory.policy = {}
            rl_memory.save()
            print("✅ Memory cleared! Starting fresh 😊\n")
            continue

        run(query)
        print("\n" + "=" * 60 + "\n")