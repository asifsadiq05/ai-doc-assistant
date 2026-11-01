"""Originality check: exact-file hash, copied-text detection (winnowing fingerprints), topic similarity."""
import hashlib
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

K, W = 5, 4            # 5-word shingles, winnowing window of 4
HIGH, PARTIAL, TOPIC = 0.70, 0.25, 0.55


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9\u0980-\u09ff]+", text.lower())  # Latin, digits and Bangla


def _h(gram: list[str]) -> int:
    return int.from_bytes(hashlib.blake2b(" ".join(gram).encode(), digest_size=8).digest(), "big")


def fingerprints(text: str) -> np.ndarray:
    t = tokens(text)
    if len(t) < K:
        return np.array([], dtype=np.uint64)
    hs = np.array([_h(t[i:i + K]) for i in range(len(t) - K + 1)], dtype=np.uint64)
    if len(hs) < W:
        return np.array([hs.min()], dtype=np.uint64)
    return np.unique(np.lib.stride_tricks.sliding_window_view(hs, W).min(axis=1))


def to_blob(a: np.ndarray) -> bytes:
    return a.astype("<u8").tobytes()


def from_blob(b) -> np.ndarray:
    return np.frombuffer(b, dtype="<u8") if b else np.array([], dtype=np.uint64)


def containment(a: np.ndarray, b: np.ndarray) -> float:
    """Share of a's fingerprints that also appear in b."""
    return float(np.isin(a, b).mean()) if len(a) and len(b) else 0.0


def topic_scores(new_text: str, texts: list[str]) -> list[float]:
    if not texts:
        return []
    try:
        X = TfidfVectorizer(stop_words="english", max_features=20000, sublinear_tf=True).fit_transform(texts + [new_text])
    except ValueError:
        return [0.0] * len(texts)
    return (X[:-1] @ X[-1].T).toarray().ravel().tolist()


def shared_passages(chunks, old_fp: np.ndarray, limit: int = 3) -> list[dict]:
    out = []
    for page, text in chunks:
        f = fingerprints(text)
        if len(f) >= 3 and containment(f, old_fp) >= 0.5:
            out.append({"page": page, "snippet": text[:200]})
        if len(out) >= limit:
            break
    return out


def analyze(new_text: str, new_fp: np.ndarray, docs: list[dict], chunks=()) -> dict:
    """docs: [{id, own, filename, fp, text}]. Other users' documents are reported anonymously."""
    matches = []
    for d, topic in zip(docs, topic_scores(new_text, [d["text"] for d in docs])):
        fp = d["fp"] if len(d["fp"]) else fingerprints(d["text"])
        ov, rev = containment(new_fp, fp), containment(fp, new_fp)
        if ov < 0.05 and topic < 0.4:
            continue
        m = {"own": d["own"], "overlap": round(ov, 3), "reverse": round(rev, 3), "topic": round(topic, 3),
             "filename": d["filename"] if d["own"] else "Another user's document"}
        if d["own"]:
            m["document_id"], m["_fp"] = d["id"], fp
        matches.append(m)
    matches.sort(key=lambda m: (m["overlap"], m["topic"]), reverse=True)
    matches = matches[:5]
    for m in matches[:3]:
        if "_fp" in m and m["overlap"] >= PARTIAL:
            m["passages"] = shared_passages(chunks, m["_fp"])
    for m in matches:
        m.pop("_fp", None)
    top = matches[0]["overlap"] if matches else 0.0
    best_topic = max((m["topic"] for m in matches), default=0.0)
    verdict = ("high" if top >= HIGH else "partial" if top >= PARTIAL
               else "topic" if best_topic >= TOPIC else "unique")
    return {"verdict": verdict, "originality": round((1 - top) * 100, 1), "top_overlap": top,
            "checked_against": len(docs), "matches": matches}
