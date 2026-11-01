"""RAG core: PDF text extraction -> chunking -> retrieval (TF-IDF or sentence-transformers)."""
import io
import logging
import os
import re

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger("docsmith.rag")
USE_ST = os.getenv("EMBEDDINGS", "tfidf").lower() == "st"
OCR = os.getenv("OCR", "0") == "1"
OCR_LANG = os.getenv("OCR_LANG", "eng")
MAX_OCR_PAGES = int(os.getenv("MAX_OCR_PAGES", "40"))
_model = None


def _ocr_page(data: bytes, index: int) -> str:
    """OCR one page. Needs: pip install pypdfium2 pytesseract + the tesseract binary."""
    import pypdfium2 as pdfium
    import pytesseract

    image = pdfium.PdfDocument(data)[index].render(scale=2).to_pil()
    return pytesseract.image_to_string(image, lang=OCR_LANG)


def extract_pages(data: bytes) -> list[tuple[int, str]]:
    reader = PdfReader(io.BytesIO(data))
    pages, ocr_used = [], 0
    for i, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text and OCR and ocr_used < MAX_OCR_PAGES:
            ocr_used += 1
            try:
                text = _ocr_page(data, i - 1).strip()
            except Exception:
                log.exception("OCR failed on page %s", i)
        if text:
            pages.append((i, text))
    return pages


def chunk_text(text: str, size: int = 900, overlap: int = 150) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        j = min(i + size, len(text))
        if j < len(text):  # break on a space, not mid-word
            s = text.rfind(" ", i + size // 2, j)
            if s != -1:
                j = s
        chunks.append(text[i:j].strip())
        if j >= len(text):
            break
        i = max(j - overlap, i + 1)
    return [c for c in chunks if len(c) > 30]


def embed(texts: list[str]) -> np.ndarray:
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return np.asarray(_model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def build_chunks(data: bytes):
    """Returns (page_count, [(page, text, embedding_bytes|None)])."""
    pages = extract_pages(data)
    items = [(p, c) for p, t in pages for c in chunk_text(t)]
    if not items:
        return len(pages), []
    vecs = embed([c for _, c in items]) if USE_ST else None
    return len(pages), [
        (p, c, vecs[i].tobytes() if vecs is not None else None) for i, (p, c) in enumerate(items)
    ]


def search(rows, query: str, k: int = 5):
    """rows: sequence of mappings with 'text' (+ 'embedding' in ST mode). Returns [(row, score)]."""
    if not rows:
        return []
    if USE_ST:
        q = embed([query])[0]
        mat = np.vstack([
            np.frombuffer(r["embedding"], dtype=np.float32) if r["embedding"] else embed([r["text"]])[0]
            for r in rows
        ])
        scores = mat @ q
    else:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        try:
            X = vec.fit_transform([r["text"] for r in rows] + [query])
        except ValueError:  # empty vocabulary
            return []
        scores = (X[:-1] @ X[-1].T).toarray().ravel()
    order = np.argsort(-scores)[:k]
    return [(rows[i], float(scores[i])) for i in order if scores[i] > 0.01]
