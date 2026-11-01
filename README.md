# Docsmith — AI Document Assistant (RAG)

A web app where users sign up, upload PDFs, and chat with them. Answers are generated from the most relevant passages and cite the file and page.

## Features
- **Originality check on every upload** (see below): exact-file, copied-text and same-topic detection against the whole database
- User authentication: register, login, logout, PBKDF2 hashing, httpOnly SameSite session cookie, hashed session tokens, CSRF header check
- PDF upload with page tracking and optional OCR for scanned PDFs (English + Bangla)
- RAG pipeline: chunk, index, retrieve top-5, generate a grounded answer that cites file and page
- Streaming answers (Server-Sent Events) from Claude; falls back to showing top passages with no API key
- Saved chat history, per-user data isolation, document scoping
- Rate limiting, security headers (CSP, nosniff, frame deny), structured logging, `/api/health`, Swagger docs at `/api/docs`
- Tests, Dockerfile (non-root, healthcheck) and GitHub Actions CI

## How the originality check works
1. **Exact duplicate:** SHA-256 of the file. Re-uploading your own file is blocked.
2. **Copied text:** word 5-grams are hashed and reduced with *winnowing* (the idea behind MOSS). The share of the new document's fingerprints found in each stored document is the overlap. Overlap of 70% or more blocks the upload until you choose "Upload anyway"; 25% or more is reported as partial.
3. **Same topic:** TF-IDF cosine similarity of whole documents flags documents on the same subject even when the wording differs.
4. **Report:** originality %, best matches, and matching passages. Click "Check" next to any document to re-run it.

Privacy: matches come from all users, but another user's document is shown only as "Another user's document" with an overlap score. Its name and text are never revealed.
Limits: topic similarity is a heuristic, and the check reads the 500 most recent documents. For a very large database, move fingerprints into an inverted index.

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY (get one at console.anthropic.com)
uvicorn app.main:app --reload
```
Open http://localhost:8000. Run tests with `pytest`. Docker: `docker build -t docsmith . && docker run -p 8000:8000 --env-file .env docsmith`.

## Better retrieval (optional)
`pip install sentence-transformers` and set `EMBEDDINGS=st` in `.env`. New uploads are embedded with all-MiniLM-L6-v2 (semantic search). Re-upload older documents after switching.

## API
| Method | Path | Purpose |
|---|---|---|
| POST | /api/register, /api/login, /api/logout | Auth (cookie). Mutating requests need `X-Requested-With: docsmith` |
| GET | /api/me | Current user |
| POST/GET | /api/documents | Upload / list PDFs |
| DELETE | /api/documents/{id} | Delete PDF and its chunks |
| POST | /api/documents?force=true | Upload; returns originality report (409 if likely copied) |
| GET | /api/documents/{id}/similarity | Originality report for a stored document |
| POST | /api/chat, /api/chat/stream | `{question, conversation_id?, document_ids?}` → answer + sources (JSON or SSE) |
| GET/DELETE | /api/conversations[/{id}] | Chat history |

## Scanned PDFs (OCR)
`pip install -r requirements-ocr.txt`, install Tesseract (`sudo apt install tesseract-ocr tesseract-ocr-ben`), then set `OCR=1` and `OCR_LANG=eng+ben`. The Docker image already includes all of this.

## Deploying
Put it behind HTTPS (Caddy or nginx), set `COOKIE_SECURE=1`, forward the real client IP for rate limiting, and mount a volume at `/srv/data`.

## Next steps
Move to Postgres + pgvector for large libraries, add email verification and password reset, and add background workers for big uploads.
