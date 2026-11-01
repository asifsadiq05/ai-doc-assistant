import json
import logging
import os
import re

from dotenv import load_dotenv
from fastapi import Cookie, Depends, FastAPI, File, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()
from . import auth, db, llm, rag, similarity  # noqa: E402  (after load_dotenv so env is set)
from .ratelimit import limit  # noqa: E402

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("docsmith")
MAX_BYTES = int(os.getenv("MAX_UPLOAD_MB", "20")) * 1024 * 1024
app = FastAPI(title="Docsmith – AI Document Assistant", docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
db.init()


@app.middleware("http")
async def security(request, call_next):
    # CSRF defence: cross-site pages cannot add this header without a CORS preflight
    if (request.method in {"POST", "PUT", "PATCH", "DELETE"} and request.url.path.startswith("/api/")
            and request.headers.get("x-requested-with") != "docsmith"):
        return JSONResponse({"detail": "Missing X-Requested-With header"}, status_code=403)
    resp = await call_next(request)
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "DENY"
    resp.headers["Referrer-Policy"] = "same-origin"
    if not request.url.path.startswith("/api/docs"):
        resp.headers["Content-Security-Policy"] = (
            "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self'; frame-ancestors 'none'")
    return resp


class RegisterIn(BaseModel):
    name: str
    email: str
    password: str


class LoginIn(BaseModel):
    email: str
    password: str


class ChatIn(BaseModel):
    question: str
    conversation_id: int | None = None
    document_ids: list[int] | None = None


@app.get("/api/health")
def health():
    return {"status": "ok", "llm": bool(os.getenv("ANTHROPIC_API_KEY")), "ocr": rag.OCR,
            "embeddings": "st" if rag.USE_ST else "tfidf"}


# ---------- auth ----------
@app.post("/api/register", dependencies=[Depends(limit("auth", 10, 60))])
def register(body: RegisterIn, response: Response):
    email = body.email.strip().lower()
    if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email):
        raise HTTPException(400, "Enter a valid email address")
    if len(body.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if not body.name.strip():
        raise HTTPException(400, "Name is required")
    salt, digest = auth.hash_password(body.password)
    with db.conn() as c:
        if c.execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise HTTPException(409, "That email is already registered")
        uid = c.execute(
            "INSERT INTO users(name,email,salt,password_hash) VALUES(?,?,?,?)",
            (body.name.strip(), email, salt, digest),
        ).lastrowid
    auth.create_session(uid, response)
    return {"name": body.name.strip()}


@app.post("/api/login", dependencies=[Depends(limit("auth", 10, 60))])
def login(body: LoginIn, response: Response):
    with db.conn() as c:
        u = c.execute("SELECT * FROM users WHERE email=?", (body.email.strip().lower(),)).fetchone()
    if not u or not auth.verify_password(body.password, u["salt"], u["password_hash"]):
        raise HTTPException(401, "Incorrect email or password")
    auth.create_session(u["id"], response)
    return {"name": u["name"]}


@app.post("/api/logout")
def logout(response: Response, session: str | None = Cookie(None)):
    auth.end_session(session, response)
    return {"ok": True}


@app.get("/api/me")
def me(user=Depends(auth.current_user)):
    return user


# ---------- originality ----------
def _report(user_id: int, exclude_id: int | None, text: str, fp, chunks) -> dict:
    with db.conn() as c:
        rows = c.execute(
            "SELECT d.id, d.user_id, d.filename, d.fingerprints, "
            "(SELECT group_concat(text, ' ') FROM chunks WHERE doc_id = d.id) AS text "
            "FROM documents d WHERE d.id != ? ORDER BY d.id DESC LIMIT 500",
            (exclude_id or 0,),
        ).fetchall()
    docs = [{"id": r["id"], "own": r["user_id"] == user_id, "filename": r["filename"],
             "fp": similarity.from_blob(r["fingerprints"]), "text": r["text"] or ""} for r in rows]
    return similarity.analyze(text, fp, docs, chunks)


# ---------- documents ----------
@app.post("/api/documents", dependencies=[Depends(limit("upload", 15, 3600))])
def upload(file: UploadFile = File(...), force: bool = False, user=Depends(auth.current_user)):
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_BYTES // 1024 // 1024} MB)")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "Only PDF files are supported")

    digest = similarity.sha256(data)
    with db.conn() as c:
        own = c.execute("SELECT filename FROM documents WHERE user_id=? AND sha256=?", (user["id"], digest)).fetchone()
    if own:
        raise HTTPException(409, {"code": "duplicate_file",
                                  "message": f'You already uploaded this exact file as "{own["filename"]}".'})
    try:
        n_pages, items = rag.build_chunks(data)
    except Exception:
        log.exception("PDF parse failed")
        raise HTTPException(400, "Could not read this PDF (it may be encrypted or corrupted)")
    if not items:
        raise HTTPException(422, "No extractable text found. Scanned PDFs need OCR (set OCR=1, see README).")

    text = " ".join(t for _, t, _ in items)
    fp = similarity.fingerprints(text)
    report = _report(user["id"], None, text, fp, [(p, t) for p, t, _ in items])
    if report["verdict"] == "high" and not force:
        raise HTTPException(409, {"code": "similar_content", "report": report,
                                  "message": "Most of this document already exists in the database."})

    with db.conn() as c:
        doc_id = c.execute(
            "INSERT INTO documents(user_id,filename,pages,n_chunks,sha256,fingerprints,n_words) VALUES(?,?,?,?,?,?,?)",
            (user["id"], file.filename or "document.pdf", n_pages, len(items), digest,
             similarity.to_blob(fp), len(similarity.tokens(text))),
        ).lastrowid
        c.executemany(
            "INSERT INTO chunks(doc_id,user_id,page,text,embedding) VALUES(?,?,?,?,?)",
            [(doc_id, user["id"], p, t, e) for p, t, e in items],
        )
    log.info("user %s uploaded doc %s (%s pages, verdict=%s)", user["id"], doc_id, n_pages, report["verdict"])
    return {"id": doc_id, "filename": file.filename, "pages": n_pages, "chunks": len(items), "report": report}


@app.get("/api/documents")
def list_documents(user=Depends(auth.current_user)):
    with db.conn() as c:
        rows = c.execute(
            "SELECT id,filename,pages,n_chunks,n_words,created_at FROM documents WHERE user_id=? ORDER BY id DESC",
            (user["id"],),
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/documents/{doc_id}/similarity")
def document_similarity(doc_id: int, user=Depends(auth.current_user)):
    with db.conn() as c:
        d = c.execute("SELECT id, fingerprints FROM documents WHERE id=? AND user_id=?", (doc_id, user["id"])).fetchone()
        if not d:
            raise HTTPException(404, "Document not found")
        chunks = [(r["page"], r["text"]) for r in
                  c.execute("SELECT page, text FROM chunks WHERE doc_id=? ORDER BY id", (doc_id,))]
    text = " ".join(t for _, t in chunks)
    fp = similarity.from_blob(d["fingerprints"]) if d["fingerprints"] else similarity.fingerprints(text)
    return _report(user["id"], doc_id, text, fp, chunks)


@app.delete("/api/documents/{doc_id}")
def delete_document(doc_id: int, user=Depends(auth.current_user)):
    with db.conn() as c:
        n = c.execute("DELETE FROM documents WHERE id=? AND user_id=?", (doc_id, user["id"])).rowcount
    if not n:
        raise HTTPException(404, "Document not found")
    return {"ok": True}


# ---------- chat ----------
def _retrieve(body: ChatIn, user: dict):
    q = body.question.strip()
    if not q:
        raise HTTPException(400, "Type a question first")
    if len(q) > 2000:
        raise HTTPException(400, "Question is too long (max 2000 characters)")
    history: list[dict] = []
    with db.conn() as c:
        if body.conversation_id:
            if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?",
                             (body.conversation_id, user["id"])).fetchone():
                raise HTTPException(404, "Conversation not found")
            history = [dict(r) for r in c.execute(
                "SELECT role,content FROM messages WHERE conv_id=? ORDER BY id", (body.conversation_id,))]
        sql = ("SELECT ch.id, ch.page, ch.text, ch.embedding, d.filename FROM chunks ch "
               "JOIN documents d ON d.id = ch.doc_id WHERE ch.user_id=?")
        params: list = [user["id"]]
        if body.document_ids:
            sql += f" AND ch.doc_id IN ({','.join('?' * len(body.document_ids))})"
            params += body.document_ids
        rows = [dict(r) for r in c.execute(sql, params).fetchall()]
    if not rows:
        return q, history, [], "Upload a PDF first, then ask me about it."
    prev = next((m["content"] for m in reversed(history) if m["role"] == "user"), "")
    query = f"{prev} {q}" if len(q.split()) < 6 and prev else q  # short follow-ups reuse the last question
    hits = rag.search(rows, query, k=5)
    if not hits:
        return q, history, [], "I couldn't find anything relevant in your documents."
    return q, history, hits, None


def _sources(hits):
    return [{"filename": r["filename"], "page": r["page"], "snippet": r["text"][:220], "score": round(s, 3)}
            for r, s in hits]


def _persist(user_id: int, conv_id: int | None, q: str, answer: str, sources: list) -> int:
    with db.conn() as c:
        if not conv_id:
            conv_id = c.execute("INSERT INTO conversations(user_id,title) VALUES(?,?)", (user_id, q[:60])).lastrowid
        c.execute("INSERT INTO messages(conv_id,role,content) VALUES(?,?,?)", (conv_id, "user", q))
        c.execute("INSERT INTO messages(conv_id,role,content,sources) VALUES(?,?,?,?)",
                  (conv_id, "assistant", answer, json.dumps(sources)))
    return conv_id


@app.post("/api/chat", dependencies=[Depends(limit("chat", 20, 60))])
def chat(body: ChatIn, user=Depends(auth.current_user)):
    q, history, hits, early = _retrieve(body, user)
    try:
        answer = early or llm.answer(q, hits, history)
    except Exception:
        log.exception("LLM call failed")
        raise HTTPException(502, "The AI service is unavailable right now. Try again in a moment.")
    sources = _sources(hits)
    return {"conversation_id": _persist(user["id"], body.conversation_id, q, answer, sources),
            "answer": answer, "sources": sources}


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.post("/api/chat/stream", dependencies=[Depends(limit("chat", 20, 60))])
def chat_stream(body: ChatIn, user=Depends(auth.current_user)):
    q, history, hits, early = _retrieve(body, user)
    sources = _sources(hits)

    def gen():
        yield _sse("meta", {"sources": sources})
        parts: list[str] = []
        try:
            for piece in ([early] if early else llm.stream_answer(q, hits, history)):
                parts.append(piece)
                yield _sse("delta", {"text": piece})
        except Exception:
            log.exception("LLM stream failed")
            yield _sse("error", {"message": "The AI service is unavailable right now. Try again in a moment."})
            return
        cid = _persist(user["id"], body.conversation_id, q, "".join(parts), sources)
        yield _sse("done", {"conversation_id": cid})

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/conversations")
def conversations(user=Depends(auth.current_user)):
    with db.conn() as c:
        rows = c.execute(
            "SELECT id,title,created_at FROM conversations WHERE user_id=? ORDER BY id DESC", (user["id"],)
        ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/conversations/{conv_id}")
def conversation(conv_id: int, user=Depends(auth.current_user)):
    with db.conn() as c:
        if not c.execute("SELECT 1 FROM conversations WHERE id=? AND user_id=?", (conv_id, user["id"])).fetchone():
            raise HTTPException(404, "Conversation not found")
        rows = c.execute("SELECT role,content,sources FROM messages WHERE conv_id=? ORDER BY id", (conv_id,)).fetchall()
    return [{"role": r["role"], "content": r["content"], "sources": json.loads(r["sources"] or "[]")} for r in rows]


@app.delete("/api/conversations/{conv_id}")
def delete_conversation(conv_id: int, user=Depends(auth.current_user)):
    with db.conn() as c:
        n = c.execute("DELETE FROM conversations WHERE id=? AND user_id=?", (conv_id, user["id"])).rowcount
    if not n:
        raise HTTPException(404, "Conversation not found")
    return {"ok": True}


# ---------- frontend ----------
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
