"""
Retrieval-Augmented Assistant
=============================

Builds a local vector store from project docs and answers questions using
OpenAI's Chat Completions API with retrieved context.

Sources used:
- README.md
- data/crop_info.json (optional)
- outputs/reports/*.pdf (optional; read as text if PyPDF2 available)

Artifacts:
- models/vector_store.joblib  (stores embeddings and index info)
- models/chunk_metadata.json  (stores chunk texts + metadata)

Environment:
- Requires OPENAI_API_KEY in .env or environment variables

CLI usage:
  python scripts/assistant.py

This will build/update the vector store from README.md and crop_info.json
and answer a sample question.
"""

from __future__ import annotations

import os
import re
import json
import glob
from pathlib import Path
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import joblib
import requests

from dotenv import load_dotenv

# Optional dependencies
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

try:
    from PyPDF2 import PdfReader  # type: ignore
    HAS_PDF = True
except Exception:
    HAS_PDF = False

try:
    from sklearn.neighbors import NearestNeighbors
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAS_TFIDF = True
except Exception:
    HAS_TFIDF = False


load_dotenv()


MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

VECTOR_STORE_PATH = MODELS_DIR / "vector_store.joblib"
METADATA_PATH = MODELS_DIR / "chunk_metadata.json"
FAISS_INDEX_PATH = MODELS_DIR / "faiss.index"
TFIDF_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"


# ---------------------------------------------------------------------------
# Helpers: reading sources
# ---------------------------------------------------------------------------
def _read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        try:
            return path.read_text(encoding="latin-1", errors="ignore")
        except Exception:
            return ""


def _read_json_file(path: Path) -> str:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except Exception:
        return ""

    def flatten(obj: Any, prefix: str = "") -> List[str]:
        rows = []
        if isinstance(obj, dict):
            for k, v in obj.items():
                rows += flatten(v, prefix=f"{prefix}{k}." if prefix else f"{k}.")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                rows += flatten(v, prefix=f"{prefix}{i}.")
        else:
            rows.append(f"{prefix[:-1]}: {obj}")
        return rows

    lines = flatten(data)
    return "\n".join(lines)


def _read_pdf_file(path: Path) -> str:
    if not HAS_PDF:
        return ""  # silently skip
    try:
        reader = PdfReader(str(path))
        texts = []
        for page in reader.pages:
            txt = page.extract_text() or ""
            texts.append(txt)
        return "\n".join(texts)
    except Exception:
        return ""


def _collect_documents(doc_paths: List[str]) -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    for pattern in doc_paths:
        for fp in glob.glob(pattern, recursive=True):
            p = Path(fp)
            if not p.exists():
                continue
            text = ""
            if p.suffix.lower() in [".md", ".txt"]:
                text = _read_text_file(p)
            elif p.suffix.lower() in [".json"]:
                text = _read_json_file(p)
            elif p.suffix.lower() == ".pdf":
                text = _read_pdf_file(p)
            if text:
                docs.append({"path": str(p), "text": text})
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def _split_into_chunks(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    """Split text by words into ~chunk_size token-equivalent chunks.

    Approximate tokens by words; keep simple and robust.
    """
    words = re.split(r"\s+", text.strip())
    chunks: List[str] = []
    if not words:
        return chunks
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if not chunk_words:
            break
        chunks.append(" ".join(chunk_words))
        i += max(1, chunk_size - overlap)
    return chunks


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def _get_openai_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set. Add it to your .env file.")
    return key


def _embed_texts(texts: List[str], model: str = "text-embedding-3-small") -> np.ndarray:
    """Call OpenAI embeddings API and return numpy array [n, d]."""
    api_key = _get_openai_api_key()
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    # OpenAI accepts up to large batch; we batch to be safe
    batch_size = 64
    all_vecs: List[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        payload = {"model": model, "input": batch}
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        vecs = [np.array(item["embedding"], dtype=np.float32) for item in data["data"]]
        all_vecs.append(np.vstack(vecs))
    return np.vstack(all_vecs) if all_vecs else np.zeros((0, 1536), dtype=np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


# ---------------------------------------------------------------------------
# Vector store build / retrieve
# ---------------------------------------------------------------------------
def build_vector_store(doc_paths: List[str]) -> Dict[str, Any]:
    """Build embeddings and index from given document path globs.

    Saves:
      - models/vector_store.joblib
      - models/chunk_metadata.json
      - models/faiss.index (if FAISS used)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    docs = _collect_documents(doc_paths)
    chunks: List[Dict[str, Any]] = []
    for d in docs:
        ctexts = _split_into_chunks(d["text"], chunk_size=350, overlap=50)
        for idx, c in enumerate(ctexts):
            if c.strip():
                chunks.append({
                    "source": d["path"],
                    "chunk_id": f"{d['path']}::chunk_{idx}",
                    "text": c,
                })

    if not chunks:
        raise RuntimeError("No text chunks found to index.")

    texts = [c["text"] for c in chunks]
    embeddings = None
    store: Dict[str, Any] = {"use_faiss": False, "faiss_path": None, "nn_index": None, "use_tfidf": False, "tfidf_path": None}

    # Try OpenAI embeddings first; fallback to TF-IDF if not available
    try:
        embeddings = _embed_texts(texts)  # [n, d]
        embeddings = _normalize_rows(embeddings)
    except Exception as e:
        if not HAS_TFIDF:
            raise RuntimeError(f"Failed to embed with OpenAI and no TF-IDF available: {e}")
        # TF-IDF fallback
        vec = TfidfVectorizer(max_features=4096)
        X = vec.fit_transform(texts)
        embeddings = X.toarray().astype(np.float32)
        embeddings = _normalize_rows(embeddings)
        joblib.dump(vec, TFIDF_PATH)
        store["use_tfidf"] = True
        store["tfidf_path"] = str(TFIDF_PATH)

    if HAS_FAISS:
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        faiss.write_index(index, str(FAISS_INDEX_PATH))
        store["use_faiss"] = True
        store["faiss_path"] = str(FAISS_INDEX_PATH)
    elif HAS_SKLEARN:
        nn = NearestNeighbors(metric='cosine', n_neighbors=10)
        nn.fit(embeddings)
        store["nn_index"] = nn
    else:
        # Fallback: store embeddings only, do brute force later
        store["nn_index"] = None

    joblib.dump({"embeddings": embeddings, **store}, VECTOR_STORE_PATH)
    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return {"num_chunks": len(chunks), "dim": embeddings.shape[1], "use_faiss": store["use_faiss"]}


def _load_store() -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
    if not VECTOR_STORE_PATH.exists() or not METADATA_PATH.exists():
        raise RuntimeError("Vector store not built. Run build_vector_store() first.")
    store = joblib.load(VECTOR_STORE_PATH)
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return store["embeddings"], store, meta


def retrieve(query: str, k: int = 4) -> List[Dict[str, Any]]:
    """Return top-k relevant chunks with metadata."""
    embeddings, store, meta = _load_store()
    if embeddings.shape[0] == 0:
        return []

    # Handle TF-IDF fallback or OpenAI embeddings
    if store.get("use_tfidf") and store.get("tfidf_path") and Path(store["tfidf_path"]).exists():
        vec = joblib.load(store["tfidf_path"])
        q_vec = vec.transform([query]).toarray().astype(np.float32)
        q_vec = _normalize_rows(q_vec)
    else:
        q_vec = _embed_texts([query])
        q_vec = _normalize_rows(q_vec)

    if store.get("use_faiss") and HAS_FAISS and store.get("faiss_path") and Path(store["faiss_path"]).exists():
        index = faiss.read_index(str(store["faiss_path"]))
        D, I = index.search(q_vec.astype(np.float32), min(k, embeddings.shape[0]))
        idxs = I[0].tolist()
        scores = D[0].tolist()
    elif store.get("nn_index") is not None and not store.get("use_tfidf"):
        nn = store["nn_index"]
        # Some joblib versions may not store index; re-fit if needed
        try:
            distances, indices = nn.kneighbors(q_vec, n_neighbors=min(k, embeddings.shape[0]))
        except Exception:
            if HAS_SKLEARN:
                re_nn = NearestNeighbors(metric='cosine', n_neighbors=min(10, embeddings.shape[0]))
                re_nn.fit(embeddings)
                store["nn_index"] = re_nn
                joblib.dump(store, VECTOR_STORE_PATH)
                distances, indices = re_nn.kneighbors(q_vec, n_neighbors=min(k, embeddings.shape[0]))
            else:
                # brute force
                sims = (embeddings @ q_vec.T).ravel()
                idxs = np.argsort(-sims)[:k].tolist()
                scores = sims[idxs].tolist()
                return [{**meta[i], "score": float(scores[j])} for j, i in enumerate(idxs)]
        idxs = indices[0].tolist()
        scores = (1.0 - distances[0]).tolist()  # cosine similarity
    else:
        sims = (embeddings @ q_vec.T).ravel()
        idxs = np.argsort(-sims)[:k].tolist()
        scores = sims[idxs].tolist()

    results = []
    for j, i in enumerate(idxs):
        if 0 <= i < len(meta):
            item = dict(meta[i])
            item["score"] = float(scores[j])
            results.append(item)
    return results


# ---------------------------------------------------------------------------
# Answer generation
# ---------------------------------------------------------------------------
def _chat_completion(messages: List[Dict[str, str]], model: str = "gpt-4o-mini") -> str:
    api_key = _get_openai_api_key()
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def answer_query(query: str, chat_history: List[Dict[str, str]] | None = None) -> Dict[str, Any]:
    """Retrieve context and answer using OpenAI Chat Completions.

    Returns a dict {"answer": str, "sources": [..]}.
    """
    chat_history = chat_history or []
    top_chunks = retrieve(query, k=4)

    context_blocks = []
    sources = []
    for c in top_chunks:
        preview = c.get("text", "")
        source = c.get("source", "")
        context_blocks.append(f"Source: {source}\n---\n{preview}")
        sources.append({"source": source, "chunk_id": c.get("chunk_id"), "score": c.get("score")})

    # If retrieval confidence is low, augment with web context (Wikipedia / SerpAPI)
    try:
        scores = [float(s.get("score", 0.0) or 0.0) for s in sources]
        top_score = max(scores) if scores else 0.0
    except Exception:
        top_score = 0.0

    if top_score < 0.7:
        web_contexts, web_srcs = _fetch_web_context(query, max_items=2)
        if web_contexts:
            context_blocks.append("\n\n".join(web_contexts))
            sources.extend(web_srcs)

    context_str = "\n\n".join(context_blocks) if context_blocks else ""
    system_prompt = (
        "You are a helpful assistant. Answer the user's question using the provided context. "
        "If the answer is not in the context, say you are not sure and suggest how to find it."
    )
    user_prompt = f"Context:\n\n{context_str}\n\nQuestion: {query}\n"

    messages = [{"role": "system", "content": system_prompt}]
    for turn in chat_history:
        if turn.get("role") in {"user", "assistant"}:
            messages.append({"role": turn["role"], "content": turn.get("content", "")})
    messages.append({"role": "user", "content": user_prompt})

    # Try OpenAI; gracefully fall back to offline if unauthorized or unavailable
    try:
        answer = _chat_completion(messages, model="gpt-4o-mini")
        return {"answer": answer, "sources": sources}
    except Exception as e:
        # Offline fallback: stitch top local chunks/web snippets
        previews = []
        for c in top_chunks:
            t = (c.get("text", "") or "")
            if t:
                previews.append(t[:500])
        stitched = ("\n\n---\n\n".join(previews)) if previews else "No local matches found."
        fallback_msg = ("OpenAI API unavailable (" + str(e) + "). \n\n" 
                        "Offline answer from local docs/web context: \n\n" + stitched)
        return {"answer": fallback_msg, "sources": sources}


# ---------------------------------------------------------------------------
# Web context fetchers (Wikipedia and optional SerpAPI)
# ---------------------------------------------------------------------------
def _fetch_web_context(query: str, max_items: int = 2) -> Tuple[List[str], List[Dict[str, Any]]]:
    """Fetch short context paragraphs from the web.

    Tries Wikipedia search + summary first, then optional SerpAPI (if SERPAPI_API_KEY set).
    Returns (contexts, sources).
    """
    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []

    # Wikipedia search
    try:
        w_ctx, w_src = _fetch_wikipedia_context(query, max_articles=max_items)
        contexts.extend(w_ctx)
        sources.extend(w_src)
    except Exception:
        pass

    # SerpAPI optional
    serp_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if serp_key and len(contexts) < max_items:
        try:
            s_ctx, s_src = _fetch_serpapi_snippets(query, max_items=max_items - len(contexts))
            contexts.extend(s_ctx)
            sources.extend(s_src)
        except Exception:
            pass

    return contexts, sources


def _fetch_wikipedia_context(query: str, max_articles: int = 2) -> Tuple[List[str], List[Dict[str, Any]]]:
    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []
    # Search
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        'action': 'query',
        'list': 'search',
        'srsearch': query,
        'format': 'json',
        'srlimit': max_articles
    }
    headers = {'User-Agent': 'SmartFarmingAssistant/1.0 (contact: support@example.com)'}
    r = requests.get(api, params=params, headers=headers, timeout=10)
    if r.status_code != 200:
        return contexts, sources
    data = r.json()
    hits = (((data or {}).get('query') or {}).get('search')) or []
    for h in hits[:max_articles]:
        title = h.get('title')
        if not title:
            continue
        # Summary
        sum_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{requests.utils.quote(title)}"
        rr = requests.get(sum_url, headers=headers, timeout=10)
        if rr.status_code != 200:
            continue
        js = rr.json()
        extract = js.get('extract') or js.get('description')
        if extract:
            contexts.append(f"Source: wikipedia:{title}\n---\n{extract}")
            sources.append({"source": f"wikipedia:{title}", "chunk_id": None, "score": None})
    return contexts, sources


def _fetch_serpapi_snippets(query: str, max_items: int = 2) -> Tuple[List[str], List[Dict[str, Any]]]:
    serp_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not serp_key:
        return [], []
    params = {
        'engine': 'google',
        'q': query,
        'api_key': serp_key,
        'num': max(1, max_items)
    }
    url = "https://serpapi.com/search"
    r = requests.get(url, params=params, timeout=15)
    if r.status_code != 200:
        return [], []
    data = r.json()
    results = (data.get('organic_results') or [])[:max_items]
    contexts: List[str] = []
    sources: List[Dict[str, Any]] = []
    for res in results:
        snippet = res.get('snippet') or res.get('title')
        link = res.get('link')
        if snippet:
            contexts.append(f"Source: web:{link}\n---\n{snippet}")
            sources.append({"source": link or "web", "chunk_id": None, "score": None})
    return contexts, sources


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        # Build (or rebuild) a minimal store from README and crop_info.json
        doc_paths = [
            str(Path("Readme.md")),
            str(Path("README.md")),  # allow either name
            str(Path("data") / "crop_info.json"),
            str(Path("outputs") / "reports" / "*.pdf"),
        ]
        info = build_vector_store(doc_paths)
        print(f"Vector store built: {info}")

        q = "What is the recommended fertilizer for rice?"
        result = answer_query(q)
        print("\nQ:", q)
        print("A:", result["answer"])
        print("Sources:")
        for s in result["sources"]:
            print(" -", s)
    except Exception as e:
        print(f"Error: {e}")


