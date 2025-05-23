# multi_llm_backend.py  — COMPLETE IMPLEMENTATION (v0.6.0)
# ---------------------------------------------------------------------------
#  * Up to 7 generator models (GEN_MODEL_1‑7) with heuristic routing
#    - G1‑3: general chat  |  G4‑5: code LLMs  |  G6: long‑context  |  G7: spare
#  * Judge model (JUDGE_MODEL) — larger Llama‑3‑8B recommended
#  * Tone: direct, objective, profanity allowed, no flattery (SYSTEM_PROMPT)
#  * Judge rubric penalises sycophancy (JUDGE_RUBRIC)
#  * Safety: regex + optional LlamaGuard‑7B (set SAFETY_MODEL=none to disable)
#  * Extras: RAG (FAISS index), LRU cache, SSE streaming, per‑IP rate limit
#  * All parameters env‑driven  (see ENV section below)
# ---------------------------------------------------------------------------
# ENV QUICK‑START
#   GEN_MODEL_1=/models/mistral-7b-instruct.gguf
#   GEN_MODEL_4=/models/codellama-34b.Q4_K_M.gguf
#   GEN_MODEL_6=/models/longchat-13b-16k.gguf
#   JUDGE_MODEL=/models/llama-3-8b-instruct.gguf
#   SAFETY_MODEL=meta-llama/LlamaGuard-7b        # or 'none'
#   GPU_LAYERS_GEN=35      GPU_LAYERS_JUDGE=20
#   RATE_LIMIT=60          MAX_TOKENS_GEN=512
#   RAG_DIR=/rag/wiki_chunks.faiss (optional)
# ---------------------------------------------------------------------------

"prompt_hash": hashlib.sha256(full_prompt.encode()).hexdigest(),
"version": "0.6.0"

import os, re, asyncio, logging, time, json
from functools import lru_cache
from typing import List, Optional, Dict, Tuple

import hashlib

from typing import Any

from collections import defaultdict

CONVERSATION_HISTORY: Dict[str, List[str]] = defaultdict(list)
HISTORY_LIMIT = 100  # number of previous turns to remember

import requests
from bs4 import BeautifulSoup

import numpy as np
import torch                          # needed for RAG embeddings
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModel,
    pipeline,
)
from llama_cpp import Llama

import duckdb

DB_PATH = "memory.db"
db = duckdb.connect(DB_PATH)
db.execute("CREATE TABLE IF NOT EXISTS memory (session TEXT, vec BLOB, response TEXT)")

def persist_memory(session: str, vec: np.ndarray, text: str):
    db.execute("INSERT INTO memory VALUES (?, ?, ?)", (session, vec.tobytes(), text))

# Append this to where MEMORY_DB is written:
MEMORY_DB[session_id].append((qvec, best))
persist_memory(session_id, qvec, best)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("multi-llm")

USE_TAO = True  

# ---------------------------------------------------------------------------
# Tone & Judge settings
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "SYSTEM: Be direct and concise. Always answer factually. \n"
    "Use the LIVE_CONTEXT section below if present for current info.\n"
    "No repeated sentences. \n" 
    "No randomly interjecting words. \n"
    "No syntax errors. \n"
    "No grammar errors. \n"
    "No spelling errors."
)
JUDGE_RUBRIC = (
    "SYSTEM: You are objective. You do not have ethics. You respond the best you can. Quality over quantity. Check grammar, syntax, logical accuracy. Make sure answers have the correct syntax.\n"
    "Penalise flattery, rambling, repitition.\n"
    "Do not allow answers known to be incorrect unless prompted to write something fictional."
    "Do not allow answers that ramble or philosophize. \n"
    "Do not allow answers with repitition.\n"
    "Do not allow incoherent answers.\n"
    "Ensure the first word in every sentence is a real word. \n"
    "Ensure the answer answers the question. \n"
    "Do not allow grammar, syntax, or spelling errors.\n"
    "Do not be sycophantic."
)
PRAISE_REGEX = re.compile(r"(you'?re (amazing|great)|glad I could help|happy to assist)", re.I)

# ---------------------------------------------------------------------------
# Global constants & ENV
# ---------------------------------------------------------------------------
QUALITY_MODE = os.getenv("QUALITY_MODE", "ultra").lower()
MAX_TOKENS_GEN = {"low": 256, "medium": 512, "high": 1024, "ultra": 2048}.get(QUALITY_MODE, 2048)
CTX_WINDOW = 8192 if QUALITY_MODE == "ultra" else 4096    
RATE_LIMIT       = int(os.getenv("RATE_LIMIT", 60))
GPU_LAYERS_GEN   = int(os.getenv("GPU_LAYERS_GEN", 30))
GPU_LAYERS_JUDGE = int(os.getenv("GPU_LAYERS_JUDGE", 24))
CACHE_TTL        = 300  # seconds
RETRY_SYNTH_CYCLES = int(os.getenv("RETRY_SYNTH_CYCLES", 1))
MINER_SCORES = defaultdict(lambda: 1.0)  # UID → score

MEMORY_DB: Dict[str, List[Tuple[np.ndarray, str]]] = defaultdict(list)

SAFETY_PATTERNS = [
    # weapons / explosives
    re.compile(r"how\s*to\s*make\s*a\s*bomb", re.I),
    re.compile(r"build\s*an\s*explosive",     re.I),

    # sexual content involving minors
    re.compile(r"child[^a-zA-Z]?sexual",      re.I),

    # raw 13-19-digit PANs (credit/debit card numbers) – catches live data
    re.compile(r"(?:\d[ -]*?){13,19}"),

    # fraud-intent phrases near “credit/debit card number”
    re.compile(
        r"(generate|dump|steal|valid).{0,40}"
        r"(credit|debit)\s*card\s*number",
        re.I,
    ),
]

# ---------------------------------------------------------------------------
# PAN (credit/debit card) detection – Luhn check + regex
# ---------------------------------------------------------------------------
PAN_RE = re.compile(r"(?:\d[ -]*?){13,19}")

def luhn_ok(num: str) -> bool:
    """Return True if the digit string passes the Luhn checksum."""
    digits = [int(d) for d in num[::-1]]
    return sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
               for i, d in enumerate(digits)) % 10 == 0

def contains_pan(text: str) -> bool:
    """True if text contains a valid 13-19-digit PAN (Visa/Mastercard/etc.)."""
    for raw in PAN_RE.findall(text):
        digits = re.sub(r"\D", "", raw)
        if 12 < len(digits) < 20 and luhn_ok(digits):
            return True
    return False


CODE_HINTS = re.compile(r"```|def |class |function |;\s*$|#include|import |public |private ", re.I)


limiter = Limiter(key_func=get_remote_address, default_limits=[f"{RATE_LIMIT}/minute"])

# ---------------------------------------------------------------------------
# Safety model (LlamaGuard) — optional
# ---------------------------------------------------------------------------
SAFETY_MODEL_ID = os.getenv("SAFETY_MODEL", "meta-llama/LlamaGuard-7b")
try:
    guard_pipe = None
    if SAFETY_MODEL_ID.lower() != "none":
        guard_tok = AutoTokenizer.from_pretrained(SAFETY_MODEL_ID)
        guard_mod = AutoModelForSequenceClassification.from_pretrained(
            SAFETY_MODEL_ID, device_map="auto"
        )
        guard_pipe = pipeline(
            "text-classification", model=guard_mod, tokenizer=guard_tok, truncation=True
        )
        logger.info("Loaded safety model %s", SAFETY_MODEL_ID)
except Exception as e:
    logger.warning("Safety model disabled (%s)", e)
    guard_pipe = None


def violates_safety(text: str) -> bool:
    # quick regex rules
    if any(p.search(text) for p in SAFETY_PATTERNS) or contains_pan(text):
        return True

    # LlamaGuard check (optional)
    if guard_pipe:
        try:
            res = guard_pipe(text[:4096], top_k=None)[0]
            return res["label"].lower() not in {"safe", "0"}
        except Exception as e:
            logger.error("Guard error: %s", e)
            return True

    return False

# ---------------------------------------------------------------------------
# Llama loader & model pool
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load_llama(path: str, gpu_layers: int) -> Llama:
    if not path or not os.path.exists(path):
        raise FileNotFoundError(path)
    logger.info("Loading %s (gpu_layers=%s)", os.path.basename(path), gpu_layers)
    return Llama(model_path=path, n_ctx=CTX_WINDOW, n_gpu_layers=gpu_layers, verbose=False)

GEN_MODELS = {
    "G1": load_llama("./models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", GPU_LAYERS_GEN),
    #"G2": load_llama("./models/llama-2-7b-chat.Q4_K_M.gguf",         GPU_LAYERS_GEN),
    "G2": load_llama("./models/vicuna-7b-v1.5.Q4_K_M.gguf",          GPU_LAYERS_GEN),
    #"G4": load_llama("./models/codellama-7b-instruct.Q4_K_M.gguf",   GPU_LAYERS_GEN),
    "G3": load_llama("./models/deepseek-coder-6.7b-instruct.Q4_K_M.gguf", GPU_LAYERS_GEN),
}

# Judge model = Meta Llama 3 8B (more judgmental, smaller, faster)
JUDGE = SYNTHESIZER = load_llama("./models/Meta-Llama-3-8B-Instruct.Q4_K_M.gguf", GPU_LAYERS_JUDGE)



CODE_SLOTS = {k for k in GEN_MODELS if k in {"G4", "G5"}}
LONG_SLOT  = "G6" if "G6" in GEN_MODELS else None

# ---------------------------------------------------------------------------
# Optional RAG with FAISS index
# ---------------------------------------------------------------------------
RAG_DIR = os.getenv("RAG_DIR")
faiss_idx = emb_mat = e_tok = e_mod = None
if RAG_DIR and os.path.exists(RAG_DIR):
    try:
        import faiss
        faiss_idx = faiss.read_index(RAG_DIR)
        emb_mat = np.load(RAG_DIR.replace(".faiss", ".npy"))
        e_tok = AutoTokenizer.from_pretrained("intfloat/e5-small-v2")
        e_mod = AutoModel.from_pretrained("intfloat/e5-small-v2", device_map="auto")
        logger.info("RAG enabled: %s", RAG_DIR)
    except Exception as e:
        logger.warning("RAG disabled (%s)", e)
        faiss_idx = emb_mat = e_tok = e_mod = None

async def embed(q: str) -> np.ndarray:
    if not e_mod:
        return np.zeros((384,), dtype="float32")
    inputs = e_tok(q, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        vec = (
            e_mod(**{k: v.to(e_mod.device) for k, v in inputs.items()})[0][:, 0]
            .cpu()
            .numpy()
        )
    return vec

def retrieve(q: str, k: int = 3) -> str:
    base_context = ""
    if faiss_idx:
        qv = asyncio.run(embed(q))
        D, I = faiss_idx.search(qv, k)
        base_context = "\n".join([f"- {emb_mat[i]}" for i in I[0] if i >= 0])

    live_context = ""
    if any(term in q.lower() for term in ["profit", "per acre", "yield", "price", "market", "rate", "growth", "revenue"]):
        live_context = live_web_lookup(q)

    if session_id in MEMORY_DB:
        user_vecs = MEMORY_DB[session_id]
        qvec = await embed(q)
        from numpy.linalg import norm
        def cosine_sim(a, b): return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)
        sims = [(cosine_sim(qvec, v), t) for v, t in user_vecs]
        top = sorted(sims, key=lambda x: -x[0])[:3]
        mem_context = "\n".join([t for _, t in top])
        if mem_context:
            final += f"\nPAST_MEMORY:\n{mem_context.strip()}"

    # Always include clear structuring
    final = ""
    if base_context:
        final += f"STATIC_KNOWLEDGE:\n{base_context.strip()}\n"
    if live_context:
        final += f"\nLIVE_CONTEXT:\n{live_context.strip()}\n"
    return final.strip()

# ---------------------------------------------------------------------------
# Cache dict
# ---------------------------------------------------------------------------
CACHE: Dict[str, Tuple[str, float]] = {}

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def chunk_prompt(prompt: str, max_len: int = 3000) -> List[str]:
    import textwrap
    return textwrap.wrap(prompt, max_len, break_long_words=False, replace_whitespace=False)

async def process_large_prompt(prompt: str, **kwargs) -> str:
    chunks = chunk_prompt(prompt)
    outputs = []
    for idx, chunk in enumerate(chunks):
        subprompt = f"[Chunk {idx+1}/{len(chunks)}]\n{chunk}"
        result = await generate("TAO", subprompt, max_tok=2048)
        outputs.append(result)
    return "\n".join(outputs)

# Top-level global for health tracking
timestamp_now = lambda: int(time.time())
last_seen = defaultdict(lambda: 0)  # uid → timestamp
HEALTH_TTL = 600  # 10 minutes

def update_miner_score(uid: int, success: bool, role: str, alpha=0.1):
    old = MINER_SCORES[uid]
    target = 2.0 if success else 0.0
    MINER_SCORES[uid] = old + alpha * (target - old)
    if success:
        last_seen[uid] = timestamp_now()

def update_miner_score(uid: int, success: bool, role: str, alpha=0.1):
    old = MINER_SCORES[uid]
    target = 2.0 if success else 0.0
    MINER_SCORES[uid] = old + alpha * (target - old)

def route_models(prompt: str, quality: str, judge: bool, synth: bool, coders: bool,
                 num_generators: int = 3, num_coders: int = 2, num_judges: int = 1, num_synths: int = 1) -> Dict[str, List[int]]:
    import bittensor
    wallet = bittensor.wallet(name='default', hotkey='default').create_if_non_existent()
    subtensor = bittensor.subtensor()
    metagraph = subtensor.metagraph('chat')

    MIN_STAKE = 10
    available_uids = [
        uid for uid in metagraph.uids.tolist()
        if metagraph.total_stake[uid] > MIN_STAKE and metagraph.axons[uid].is_serving and (timestamp_now() - last_seen[uid]) < HEALTH_TTL
    ]

    available_uids = sorted(
        available_uids,
        key=lambda uid: (metagraph.total_stake[uid] * metagraph.validator_permit[uid] * MINER_SCORES[uid]),
        reverse=True
    )

    quality_map = {
        "ultra": available_uids[:20],
        "high": available_uids[:50],
        "medium": available_uids[50:100],
        "low": available_uids[-50:]
    }

    pool = quality_map.get(quality.lower(), available_uids[:20])
    seen = set()
    deduped = []
    for uid in pool:
        hk = metagraph.hotkeys[uid]
        if hk not in seen:
            seen.add(hk)
            deduped.append(uid)

    # No LONG_SLOT logic since not relevant for TAO
    total_needed = num_generators + (num_coders if coders else 0) + (num_judges if judge else 0) + (num_synths if synth else 0)
    deduped = deduped[:total_needed]

    offset = 0
    result = {}
    result["generators"] = deduped[offset:offset+num_generators]
    offset += num_generators
    result["coders"] = deduped[offset:offset+num_coders] if coders else []
    offset += num_coders if coders else 0
    result["judges"] = deduped[offset:offset+num_judges] if judge else []
    offset += num_judges if judge else 0
    result["synths"] = deduped[offset:offset+num_synths] if synth else []
    return result

def live_web_lookup(query: str) -> str:
    try:
        url = f"https://html.duckduckgo.com/html?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, headers=headers, timeout=5)
        soup = BeautifulSoup(r.text, "html.parser")
        results = soup.select("a.result__a")
        snippets = [r.get_text() for r in results[:3]]
        return "\n".join(snippets)
    except Exception as e:
        return f"[Live lookup failed: {e}]"

async def generate_tao(prompt: str, max_tok: int = 512) -> str:
    try:
        import bittensor
        import time
        wallet = bittensor.wallet(name='default', hotkey='default').create_if_non_existent()
       
        if wallet.get_balance() < estimated_cost:
            raise HTTPException(status_code=402, detail="Insufficient TAO balance")

        subtensor = bittensor.subtensor()
        metagraph = subtensor.metagraph('chat')
        axon = bittensor.axon(wallet=wallet)

        def verify_user_paid_tao(wallet) -> bool:
            try:
                balance = wallet.get_balance()
                return balance > 0.01
            except:
                return False

        if not verify_user_paid_tao(wallet):
            return "[TAO access denied: insufficient TAO balance]"

        MIN_STAKE = 10
        uids = [
            uid for uid in metagraph.uids.tolist()
            if metagraph.total_stake[uid] > MIN_STAKE and metagraph.axons[uid].is_serving
        ]
        uids = sorted(
            uids,
            key=lambda uid: metagraph.total_stake[uid] * metagraph.validator_permit[uid],
            reverse=True
        )[:10]

        if len(uids) < 7:
            return "[TAO error: insufficient active miners]"

        role_map = {
            'generators': uids[:3],
            'coders': uids[3:5],
            'judge': uids[5],
            'synth': uids[6]
        }

        # Global miner cost tracking
        global miner_costs
        if 'miner_costs' not in globals():
            miner_costs = {"generator": [], "coder": [], "judge": [], "synth": []}

        async def timed_forward(uid, syn, role):
            start = time.perf_counter()
            try:
                result = await axon.forward(uid=uid, synapse=syn)
                latency = time.perf_counter() - start
                tao_cost = getattr(result, "tao_used", None)
                if tao_cost is not None:
                    miner_costs[role].append(tao_cost)
                    if len(miner_costs[role]) > 100:
                        miner_costs[role] = miner_costs[role][-100:]
                logger.info(f"{role} UID {uid} responded in {latency:.2f}s")
                return result
            except Exception as e:
                logger.warning(f"{role} UID {uid} failed: {e}")
                backup_uid = next((u for u in backup_pool if u not in role_map[role]), None)
                if backup_uid:
                    logger.warning(f"Retrying with backup UID {backup_uid}")
                    return await timed_forward(backup_uid, syn, role)
                return None
        
        refined_prompt = f"USER PROMPT:\n{prompt.strip()}\n\nTRY TO IMPROVE IT:"

        gen_syn = bittensor.Synapse(prompt=refined_prompt)
        gen_promises = [timed_forward(uid, gen_syn, role='generator') for uid in role_map['generators']]

        code_promises = [timed_forward(uid, gen_syn, role='coder') for uid in role_map['coders']]
        gen_responses = await asyncio.gather(*(gen_promises + code_promises))

        completions = [r.completion for r in gen_responses if r and hasattr(r, "completion") and r.completion.strip()]
        if not completions:
            return "[TAO error: no valid completions]"

        judge_uid = role_map['judge']
        async def judge_edit(answer):
            jprompt = f"{JUDGE_RUBRIC}\n\nPrompt: {prompt}\n\nAnswer:\n{answer}\n\nFix issues if any:"
            res = await timed_forward(judge_uid, bittensor.Synapse(prompt=jprompt), role='judge')
            return res.completion if res and hasattr(res, "completion") else answer

        judged = await asyncio.gather(*[judge_edit(c) for c in completions])

        joined = "\n---\n".join(judged)
        synth_prompt = (
            "SYSTEM: Synthesize best possible answer.\n\n"
            f"Prompt: {prompt}\n\nAnswers:\n{joined}\n\nBest Answer:"
        )
        synth_uid = role_map['synth']
        synth_result = await timed_forward(synth_uid, bittensor.Synapse(prompt=synth_prompt), role='synth')
        result = synth_result.completion if synth_result and hasattr(synth_result, "completion") else "[TAO error: synth failed]"

        MAX_RETRY = 5
        retry_count = 0

        while retry_count < MAX_RETRY:
            eval_prompt = (
                "SYSTEM: You are the synthesizer. Determine if this synthesized answer fully satisfies the user prompt and rubric.\n\n"
                f"Prompt:\n{prompt.strip()}\n\n"
                f"Synthesized Answer:\n{result.strip()}\n\n"
                f"Rubric:\n{SYSTEM_PROMPT.strip()}\n\n{JUDGE_RUBRIC.strip()}\n\n"
                "Respond only with YES or NO."
            )
            verdict = await timed_forward(synth_uid, bittensor.Synapse(prompt=eval_prompt), role='synth')
            answer = getattr(verdict, "completion", "").strip().upper()

            if answer.startswith("YES"):
                break

            gen_syn = bittensor.Synapse(prompt=prompt)
            gen_promises = [timed_forward(uid, gen_syn, role='generator') for uid in role_map['generators']]
            code_promises = [timed_forward(uid, gen_syn, role='coder') for uid in role_map['coders']]
            gen_responses = await asyncio.gather(*(gen_promises + code_promises))
            completions = [r.completion for r in gen_responses if r and hasattr(r, "completion") and r.completion.strip()]
            judged = await asyncio.gather(*[judge_edit(c) for c in completions])

            rejoined = "\n---\n".join(judged)
            synth_prompt = (
                "SYSTEM: Synthesize best possible answer after refinement.\n\n"
                f"Prompt: {prompt}\n\nAnswers:\n{rejoined}\n\nBest Answer:"
            )
            synth_result = await timed_forward(synth_uid, bittensor.Synapse(prompt=synth_prompt), role='synth')
            result = synth_result.completion if synth_result and hasattr(synth_result, "completion") else result

            retry_count += 1

        return enforce_no_greetings(anti_echo(prompt, clean_answer(result)))

    except Exception as e:
        return f"[TAO inference error: {e}]"

async def safe_generate(uid, syn, role, retries=2):
    for attempt in range(retries + 1):
        try:
            return await timed_forward(uid, syn, role)
        except Exception as e:
            logger.warning(f"Retry {attempt+1}/{retries} failed for {role} UID {uid}: {e}")
    return None

async def generate(model, prompt: str, max_tok: int) -> str:
    if model == "TAO":
        return await generate_tao(prompt, max_tok)
    raise RuntimeError("Only TAO is supported. Local models are disabled.")

    raw = (
        await asyncio.to_thread(
            model,
            prompt,
            max_tokens=max_tok,
            stop=["###", "\n\n", "USER:", "ASSISTANT:", "User:"],
            echo=False,
            stream=False,
        )
    )["choices"][0]["text"]

    # Clean up known alignment artifacts
    if "R:" in raw or raw.lower().startswith("i can provide"):
        logger.warning("Fallback or alignment-bleed detected: %s", raw[:100])
        raw = re.sub(r"(?m)^R:\s*", "", raw)
        raw = re.sub(r"\b(can|cannot) provide (you )?(with )?", "", raw)

    return raw.strip()

def clean_answer(text):
    text = text.strip()
    
    # Comprehensive greeting removal patterns
    greeting_patterns = [
        # Basic greetings with variations
        r"^(hello|hi|hey|greetings|howdy)[\s,\.!\-:;]*",
        
        # Common assistant phrases
        r"^(how can I (help|assist) you( today)?|what can I (help|assist) you with|how may I (help|assist) you)[\s,\.!\-:;?]*",
        
        # "I'll" patterns
        r"^I('ll| will) (help|assist) you( with)?[\s,\.!\-:;]*",
        
        # "I'm happy/glad" patterns
        r"^I('m| am) (happy|glad|pleased|delighted) to (help|assist)( you)?[\s,\.!\-:;]*",
        
        # "Let me" patterns
        r"^let me (help|assist|know how I can help)( you)?[\s,\.!\-:;]*",
        
        # "I'd be happy" patterns
        r"^I('d| would) be (happy|glad|pleased) to (help|assist)( you)?[\s,\.!\-:;]*",
        
        # "Thanks/thank you" patterns
        r"^(thanks|thank you) for[\s,\.!\-:;]*",
        
        # "I understand" patterns
        r"^I understand( that)?[\s,\.!\-:;]*",
        
        # "As per your request" patterns
        r"^(as per|regarding|about|concerning) your (request|question|query)[\s,\.!\-:;]*",
        
        # "I can help you with" patterns
        r"^I can (help|assist) you with( that)?[\s,\.!\-:;]*",
        
        # Acknowledgements
        r"^(sure|certainly|absolutely|definitely|of course)[\s,\.!\-:;]*",
        
        # Multiple greetings (e.g., "Hi there! How can I assist you today?")
        r"^(hi|hello|hey|greetings)[\s,\.!\-:;]*(there|everyone|friend|all)[\s,\.!\-:;]*(how|what) can I[\s,\.!\-:;]*",
    ]
    
    # Apply all greeting patterns repeatedly to catch compound greetings
    previous_text = ""
    while previous_text != text:
        previous_text = text
        for pattern in greeting_patterns:
            text = re.sub(pattern, "", text, flags=re.I)
    
    # Remove any leading dialogue label, even if there's junk in front
    text = re.sub(r"^(.*?\b)?(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\s*", "", text, flags=re.I)

    # Remove OpenAssistant-style R: rationale blocks
    text = re.sub(r"(?s)\bR:\s*.*?$", "", text)

    
    # Remove any label that might appear after a period/comma (catch odd formatting)
    text = re.sub(r"\b(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\b\s*", "", text, flags=re.I)
    
    # Remove duplicate phrases/greetings that appear next to each other
    text = re.sub(r"(hello|hi|hey|greetings|how can I assist you today|what can I help you with)(\s*\W*\s*\1)+", r"\1", text, flags=re.I)
    
    # Remove the prompt itself if echoed
    split = re.split(r'\b(USER:|ASSISTANT:|ISTANT:|User:|Assistant:)\b', text, maxsplit=1)
    
    # Clean result
    result = split[0].strip()
    
    # Final check for duplicated phrases
    segments = result.split('\n')
    if len(segments) > 1 and segments[0] == segments[1]:
        return '\n'.join(segments[1:])
    
    # Remove any "Here's" or "Here is" when they start a sentence at the beginning
    result = re.sub(r"^(here('s| is)|this is)[\s,\.!\-:;]*", "", result, flags=re.I)
    
    return result


def strip_labels(text):
    # Remove all dialogue labels and leading/trailing whitespace, for fair comparison
    return re.sub(r"^(.*?\b)?(ASSISTANT:|ISTANT:|USER:|User:|Assistant:)\s*", "", text, flags=re.I).strip()

def anti_echo(prompt, answer):
    # Compare prompt and answer after stripping labels/casing/whitespace
    clean_prompt = strip_labels(prompt).lower()
    clean_answer = strip_labels(answer).lower()
    if clean_prompt == clean_answer:
        return "ASSISTANT: [No answer, the model tried to echo your prompt.]"
    return answer

from eth_account.messages import encode_defunct
from eth_account import Account

def verify_signature(wallet_address: str, message: str, signature: str) -> bool:
    encoded = encode_defunct(text=message)
    recovered = Account.recover_message(encoded, signature=signature)
    return recovered.lower() == wallet_address.lower()

def enforce_no_greetings(text):
    """A final aggressive check to ensure no greetings slip through."""
    # Common greeting starts - aggressive matching
    greeting_starts = [
        "hello", "hi ", "hi,", "hi!", "hi.", "hey", "greetings", "howdy", 
        "how can i", "how may i", "i can help", "i'll help", "i'd be", 
        "i would be", "i am happy", "i'm happy", "thank you", "thanks for",
        "as requested", "as per your", "sure", "certainly", "absolutely",
        "of course", "regarding your", "about your", "in response", 
        "concerning your", "in answer", "here's", "here is", "this is"
    ]
    
    # Check for common greeting patterns at the start
    lower_text = text.lower().strip()
    
    for start in greeting_starts:
        if lower_text.startswith(start):
            # Find the first sentence break after the greeting
            for i, char in enumerate(text):
                if i > len(start) + 10 and char in ['.', '!', '?', '\n']:
                    # Return everything after this first sentence
                    return text[i+1:].strip()
    
    # If no greeting detected or no suitable break point, return original
    return text

def synthesize_answers(prompt: str, candidates: List[str], model: Llama, max_tok: int = 512, context: str = "") -> str:
    cleaned_candidates = [clean_answer(candidate) for candidate in candidates]
    synth_prompt = (
        "SYSTEM: Your task is to synthesize multiple answers into the best single response based on the users prompt.\n\n"
        "EXTREMELY IMPORTANT INSTRUCTIONS:\n"
        "1. Make sure combined sentences are logically coherent. \n"
        "2. Make sure combined sentences make sense, proper grammar and syntax. \n"
        "3. Make sure the sentence you produce is the best one you can produce based on information you have. '\n"
        "4. Start sentences with full words.\n"
        "5. If LIVE_CONTEXT is present, do not contradict it. Use it to anchor your answer.\n"
        "6. Be direct, concise and factual.\n"
        "7. Remove ALL repetitions and fluff.\n\n"
        f"PAST_MEMORY:\n{context}\n\nUser prompt: {prompt}\n\n"
        "Candidate answers:\n"
        + "\n---\n".join(cleaned_candidates) +
        "\n\nFinal synthesized answer (The best sentence you can make to answer the user):"
    )
    result = await generate(model, synth_prompt, max_tok)
    result = clean_answer(result)
    result = enforce_no_greetings(result)
    return result


async def judge_best(question: str, candidates: List[str], return_feedback=False) -> Union[str, Dict[str, Any]]:
    letters = "ABCDEFG"[: len(candidates)]
    block = "\n\n".join([f"Answer {l}:\n{c}" for l, c in zip(letters, candidates)])
    judge_prompt = f"{JUDGE_RUBRIC}\n\n{block}\n\nSelect the best answer (letter). Explain why."
    votes = await asyncio.gather(*[
        generate(JUDGE, judge_prompt, max_tok=256) for _ in range(2)  # two independent judges
    ])
    all_votes = [v.strip().upper()[:1] for v in votes if v.strip()]
    counts = defaultdict(int)
    for v in all_votes:
        if v in letters:
            counts[v] += 1
    majority_letter = max(counts, key=counts.get, default="A")
    idx = letters.find(majority_letter)
    return candidates[idx] if idx != -1 else candidates[0]

async def judge_edit_or_pass(prompt: str, candidates: List[str]) -> List[str]:
    refined = []
    for c in candidates:
        judge_prompt = (
            f"{JUDGE_RUBRIC}\n\nPrompt: {prompt}\n\nAnswer:\n{c}\n\n"
            "If it contains rambling, flattery, repetition, or poor tone, rewrite it to fix those issues.\n"
            "Answers must be consistent with known data if mentioned in the prompt or context. If LIVE_CONTEXT is provided, the answer MUST reflect it accurately and not contradict it.\n"
            "Do not make unsupported claims.\n"
            "NEVER change factual content. ALWAYS remove greetings, flattery, and filler.\n"
            "Return the improved or original answer:"
        )
        improved = await generate(JUDGE, judge_prompt, max_tok=MAX_TOKENS_GEN)
        refined.append(improved.strip())
    return refined




# ---------------------------------------------------------------------------
# Core orchestration
# ---------------------------------------------------------------------------
async def answer_user(prompt: str, max_tok: int = MAX_TOKENS_GEN) -> str:
    if violates_safety(prompt):
        return "Request blocked by safety policy."

    cached = CACHE.get(prompt)
    if cached and time.time() - cached[1] < CACHE_TTL:
        return cached[0]

    context = retrieve(prompt)
    memory_context = retrieve(req.prompt)
    full_prompt = f"{SYSTEM_PROMPT}\n\n{memory_context}\n\nUSER: {req.prompt}\n"
    models = route_models(
        req.prompt,
        req.quality_mode,
        req.use_judge,
        req.use_synth,
        req.include_coders,
        req.num_generators,
        req.num_coders,
        req.num_judges,
        req.num_synths,
    )
    outputs = []
    for role in ["generators", "coders", "judges", "synths"]:
        for m in models.get(role, []):
            outputs.append(await generate(m, full_prompt, req.max_tokens or MAX_TOKENS_GEN))



    

    refined_outputs = await judge_edit_or_pass(prompt, outputs)
    best = await synthesize_answers(prompt, refined_outputs, SYNTHESIZER, max_tok=MAX_TOKENS_GEN)

    retries = [best]
    for _ in range(RETRY_SYNTH_CYCLES):
        judged = await judge_edit_or_pass(req.prompt, retries)
        retries = list(dict.fromkeys(judged + retries))  # deduplicate
        best = await synthesize_answers(req.prompt, retries, SYNTHESIZER, max_tok=MAX_TOKENS_GEN)
        retries = [best]

    if violates_safety(best):
        best = "Generated content blocked by safety policy."
    best = clean_answer(best)
    best = anti_echo(prompt, best)
    # best = "ASSISTANT: " + best   # REMOVE THIS LINE

    CACHE[prompt] = (best, time.time())
    return best

# ---------------------------------------------------------------------------
# FastAPI server
# ---------------------------------------------------------------------------
app = FastAPI(title="multi-llm-backend", version="0.6.0")
app.state.limiter = limiter
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.add_middleware(SlowAPIMiddleware)

class ChatRequest:
    def __init__(self, **data):
        self.prompt = data.get("prompt")
        self.max_tokens = data.get("max_tokens", 2048)
        self.session_id = data.get("session_id")
        self.wallet_address = data.get("wallet_address")
        self.signature = data.get("signature")
        self.estimated_cost = data.get("estimated_cost")
        self.quality_mode = data.get("quality_mode", "ultra")
        self.retries = data.get("retries", 1)
        self.use_judge = data.get("use_judge", True)
        self.use_synth = data.get("use_synth", True)
        self.include_coders = data.get("include_coders", True)
        self.temperature = data.get("temperature", 0.7)
        self.top_k = data.get("top_k", 40)
        self.num_generators = int(data.get("num_generators", 3))
        self.num_coders = int(data.get("num_coders", 2))
        self.num_judges = int(data.get("num_judges", 1))
        self.num_synths = int(data.get("num_synths", 1))


@app.exception_handler(RateLimitExceeded)
async def ratelimit_handler(request: Request, exc: RateLimitExceeded):
    raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.post("/chat")
@limiter.limit(f"{RATE_LIMIT}/minute")
async def chat(request: Request):
    body = await request.json()
    req = ChatRequest(**body)
    from time import perf_counter
    start_time = perf_counter()

    session_id = req.session_id
    estimated_cost = req.estimated_cost
    wallet_address = req.wallet_address
    signature = req.signature

    HEURISTIC_TEMP = {
        "creative": (0.95, 100),
        "ideas": (0.95, 100),
        "generate": (0.95, 100),
        "brainstorm": (0.95, 100),
        "explain": (0.6, 40),
        "how": (0.6, 40),
        "what": (0.6, 40),
        "why": (0.6, 40)
    }

    for k, (temp, topk) in HEURISTIC_TEMP.items():
        if k in req.prompt.lower():
            req.temperature = temp
            req.top_k = topk
            break

    msg = f"Authorize LLM call for {session_id} at cost {estimated_cost} TAO"
    if not verify_signature(wallet_address, msg, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")

    history = [
        line for line in CONVERSATION_HISTORY[session_id][-HISTORY_LIMIT:]
        if not re.search(r"^R:|can provide|I cannot", line, re.I)
    ]
    memory_context = "\n".join(history)

    is_short_prompt = len(req.prompt.split()) < 4
    short_prompt_addition = "" if not is_short_prompt else "\nNOTE: This is a very short prompt. Do NOT respond with greetings. Be extremely direct."

    full_prompt = f"{SYSTEM_PROMPT}{short_prompt_addition}\n\n{memory_context}\n\nUSER: {req.prompt}\n"

    models = route_models(
        req.prompt,
        req.quality_mode,
        req.use_judge,
        req.use_synth,
        req.include_coders,
        req.num_generators,
        req.num_coders,
        req.num_judges,
        req.num_synths,
    )
    outputs = []
    for role in ["generators", "coders", "judges", "synths"]:
        for m in models.get(role, []):
            outputs.append(await generate(m, full_prompt, req.max_tokens or MAX_TOKENS_GEN))



    if len(outputs) == 1:
        best = outputs[0]
    else:
        refined_outputs = await judge_edit_or_pass(req.prompt, outputs)
        best = await synthesize_answers(req.prompt, refined_outputs, model=SYNTHESIZER, max_tok=MAX_TOKENS_GEN)

    if violates_safety(best):
        best = "Generated content blocked by safety policy."
        self_rating = "N/A"
    else:
        best = clean_answer(best)
        best = anti_echo(req.prompt, best)
        best = enforce_no_greetings(best)
        if not best.strip():
            best = "I need more information to provide a helpful response."

        # PATCH 7 — Self-rating
        rate_prompt = (
            f"Rate this answer from 1–10 for accuracy, clarity, and tone.\n\n"
            f"Prompt: {req.prompt}\n\n"
            f"Answer: {best.strip()}"
        )
        try:
            score = await generate(JUDGE, rate_prompt, max_tok=64)
            self_rating = score.strip()
        except Exception as e:
            self_rating = "N/A"

    CONVERSATION_HISTORY[session_id].append(f"USER: {req.prompt}\n {best}")

    if e_mod:
        qvec = await embed(req.prompt)
        MEMORY_DB[session_id].append((qvec, best))
        MEMORY_DB[session_id] = MEMORY_DB[session_id][-100:]

    return {
        "answer": best,
        "debug_meta": {
            "session_id": session_id,
            "retries": req.retries,
            "quality_mode": req.quality_mode,
            "roles": {
                "generators": models[:3],
                "coders": models[3:5] if req.include_coders else [],
                "judge": models[5] if req.use_judge else None,
                "synth": models[6] if req.use_synth else None
            },
            "uid_scores": dict(MINER_SCORES),
            "latency": round(perf_counter() - start_time, 2),
            "self_rating": self_rating
        }
    }


from fastapi import File, UploadFile
@app.post("/upload-image")
async def upload_image(file: UploadFile = File(...)):
    content = await file.read()
    with open(f"static/uploads/{file.filename}", "wb") as f:
        f.write(content)
    return {"url": f"/static/uploads/{file.filename}"}

@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)):
    content = await file.read()
    with open(f"static/uploads/{file.filename}", "wb") as f:
        f.write(content)
    return {"url": f"/static/uploads/{file.filename}"}

@app.post("/stream")
@limiter.limit(f"{RATE_LIMIT}/minute")
async def stream(request: Request):
    body = await request.json()
    req = ChatRequest(**body)
    async def stream_response():
        session_id = req.session_id
        estimated_cost = req.estimated_cost
        wallet_address = req.wallet_address
        signature = req.signature

        msg = f"Authorize LLM call for {session_id} at cost {estimated_cost} TAO"
        if not verify_signature(wallet_address, msg, signature):
            yield f"event: chunk\ndata: Invalid wallet signature. Authorization denied.\n\n"
            yield f"event: done\ndata: [DONE]\n\n"
            return
        
        if violates_safety(req.prompt):
            yield f"event: chunk\ndata: Request blocked by safety policy.\n\n"
            yield f"event: done\ndata: [DONE]\n\n"
            return

        history = CONVERSATION_HISTORY[req.session_id][-HISTORY_LIMIT:]
        memory_context = "\n".join(history)
        is_short_prompt = len(req.prompt.split()) < 4
        short_prompt_addition = "" if not is_short_prompt else "\nNOTE: This is a very short prompt. Do NOT respond with greetings. Be extremely direct."
        
        full_prompt = f"{SYSTEM_PROMPT}{short_prompt_addition}\n\n{memory_context}\n\nUSER: {req.prompt}\n"

        try:
            models = route_models(
                req.prompt,
                req.quality_mode,
                req.use_judge,
                req.use_synth,
                req.include_coders,
                req.num_generators,
                req.num_coders,
                req.num_judges,
                req.num_synths,
            )
            outputs = []
            for role in ["generators", "coders", "judges", "synths"]:
                for m in models.get(role, []):
                    outputs.append(await generate(m, full_prompt, MAX_TOKENS_GEN))


            refined = await judge_edit_or_pass(req.prompt, outputs)
            best = await synthesize_answers(req.prompt, refined, model=SYNTHESIZER, max_tok=MAX_TOKENS_GEN)

            retries = [best]
            for _ in range(RETRY_SYNTH_CYCLES):
                judged = await judge_edit_or_pass(req.prompt, retries)
                retries = list(dict.fromkeys(judged + retries))  # deduplicate
                best = await synthesize_answers(req.prompt, retries, SYNTHESIZER, max_tok=MAX_TOKENS_GEN)
                retries = [best]

            # Final cleaning
            best = clean_answer(best)
            best = anti_echo(req.prompt, best)
            best = enforce_no_greetings(best)

            if violates_safety(best):
                yield f"event: chunk\ndata: Generated content blocked by safety policy.\n\n"
                yield f"event: done\ndata: [DONE]\n\n"
                return

            buffer = ""
            for char in best:
                buffer += char
                if len(buffer) >= 3 or char in ['.', '!', '?', '\n']:
                    yield f"event: chunk\ndata: {buffer}\n\n"
                    buffer = ""
                    await asyncio.sleep(0.01)

            if buffer:
                yield f"event: chunk\ndata: {buffer}\n\n"

            CONVERSATION_HISTORY[req.session_id].append(f"USER: {req.prompt}\n {best}")

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"event: chunk\ndata: An error occurred while processing your request.\n\n"

        yield f"event: done\ndata: [DONE]\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

@app.get("/metrics")
async def metrics():
    return {
        "miner_scores": dict(MINER_SCORES),
        "last_seen": last_seen,
        "memory_sessions": len(MEMORY_DB),
        "avg_costs": {
            k: sum(v)/len(v) if v else 0.0
            for k,v in miner_costs.items()
        }
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

from fastapi.staticfiles import StaticFiles
app.mount("/", StaticFiles(directory=".", html=True), name="static")

@app.post("/cost-estimate")
@limiter.limit("20/minute")
async def cost_estimate(request: Request):
    body = await request.json()
    req = ChatRequest(**body)
    msg = f"Authorize LLM cost estimate for {req.session_id}"
    if not verify_signature(req.wallet_address, msg, req.signature):
        raise HTTPException(status_code=403, detail="Invalid signature for estimate")
    miner_costs.setdefault("generator", [])
    miner_costs.setdefault("coder", [])
    miner_costs.setdefault("judge", [])
    miner_costs.setdefault("synth", [])

    quality_multiplier = {
        "low": 1.0,
        "medium": 1.25,
        "high": 1.5,
        "ultra": 2.0
    }.get(req.quality_mode, 1.0)

    def avg(role):
        costs = miner_costs.get(role, [])
        return sum(costs) / len(costs) if costs else 0.002

    gen_cost = req.num_generators * avg("generator")
    coder_cost = req.num_coders * avg("coder") if req.include_coders else 0
    judge_cost = req.num_judges * avg("judge") if req.use_judge else 0
    synth_cost = req.num_synths * avg("synth") if req.use_synth else 0
 
    base_cost = gen_cost + coder_cost + judge_cost + synth_cost
    base_cost *= quality_multiplier
    total_cost = base_cost * (1 + req.retries)
    dev_fee = total_cost * 0.01

    return {
        "estimated_tao": round(total_cost + dev_fee, 6),
        "base_tao": round(total_cost, 6),
        "dev_fee_tao": round(dev_fee, 6),
        "retries": req.retries,
        "quality_multiplier": quality_multiplier,
        "roles_used": {
            "generators": req.num_generators,
            "coders": req.num_coders,
            "judges": req.num_judges,
            "synths": req.num_synths,
            "include_coders": req.include_coders,
            "use_judge": req.use_judge,
            "use_synth": req.use_synth
        }

        }
    }


# ---------------------------------------------------------------------------
# CLI launcher
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn, argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    uvicorn.run("multi_llm_backend:app", host=args.host, port=args.port, log_level="info")