import io
import os
import tempfile

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "t.db")
os.environ["RATE_LIMIT"] = "0"
os.environ.pop("ANTHROPIC_API_KEY", None)

from fastapi.testclient import TestClient  # noqa: E402

import pytest  # noqa: E402

from app import db, ratelimit, rag, similarity  # noqa: E402
from app.main import app  # noqa: E402

H = {"X-Requested-With": "docsmith"}


@pytest.fixture(autouse=True)
def clean_db():
    with db.conn() as c:  # every test starts with an empty database
        for t in ("messages", "conversations", "chunks", "documents", "sessions", "users"):
            c.execute(f"DELETE FROM {t}")



def new_client() -> TestClient:
    return TestClient(app, headers=H)


def make_pdf(text: str) -> bytes:
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        "/Resources << /Font << /F1 5 0 R >> >> >>",
    ]
    stream = f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET"
    objs.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objs.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    out, offs = b"%PDF-1.4\n", []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f"{i} 0 obj\n{o}\nendobj\n".encode()
    xref = len(out)
    out += f"xref\n0 {len(objs)+1}\n0000000000 65535 f \n".encode()
    for o in offs:
        out += f"{o:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs)+1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF".encode()
    return out


def upload(client, text, name="a.pdf", force=False):
    return client.post("/api/documents" + ("?force=true" if force else ""),
                       files={"file": (name, io.BytesIO(make_pdf(text)), "application/pdf")})


def register(client, email):
    r = client.post("/api/register", json={"name": "T", "email": email, "password": "password1"})
    assert r.status_code == 200
    return r


NETWORKS = ("Computer networks move data between machines using layered protocols such as TCP and IP. "
            "Routers forward packets along paths chosen by routing algorithms while switches connect devices "
            "inside a local network. Congestion control in TCP slows senders when queues grow and the network "
            "becomes overloaded so that delivery stays fair and reliable for every connected application.")
COOKING = ("To bake sourdough bread you need flour water salt and a lively starter. Mix the dough and let it "
           "ferment overnight in a cool kitchen then shape it and bake in a very hot covered pot until the "
           "crust turns deep brown and crackles when it cools on a wire rack near the open window.")


def test_chunking_and_fingerprints():
    assert all(len(c) <= 900 for c in rag.chunk_text("word " * 1000))
    assert similarity.containment(similarity.fingerprints(NETWORKS), similarity.fingerprints(NETWORKS)) == 1.0
    assert similarity.containment(similarity.fingerprints(NETWORKS), similarity.fingerprints(COOKING)) < 0.05


def test_auth_cookie_and_csrf():
    c = new_client()
    assert c.get("/api/documents").status_code == 401
    r = register(c, "auth@x.com")
    assert "httponly" in r.headers["set-cookie"].lower() and "samesite=lax" in r.headers["set-cookie"].lower()
    assert c.get("/api/me").json()["email"] == "auth@x.com"
    no_header = TestClient(app, cookies=c.cookies)  # a cross-site request cannot add the header
    assert no_header.post("/api/logout").status_code == 403
    assert c.post("/api/logout").status_code == 200
    assert c.get("/api/me").status_code == 401
    assert c.post("/api/login", json={"email": "nobody@x.com", "password": "whatever1"}).status_code == 401


def test_rag_chat_history_and_isolation():
    c = new_client(); register(c, "rag@x.com")
    assert upload(c, NETWORKS, "net.pdf").status_code == 200
    assert upload(c, "nope", "bad.pdf").status_code in (200, 400, 422)  # valid PDF, tiny text
    r = c.post("/api/chat", json={"question": "How does TCP congestion control work?"})
    body = r.json()
    assert r.status_code == 200 and body["sources"][0]["filename"] == "net.pdf"
    msgs = c.get(f"/api/conversations/{body['conversation_id']}").json()
    assert [m["role"] for m in msgs] == ["user", "assistant"]

    other = new_client(); register(other, "rag2@x.com")
    assert other.get("/api/documents").json() == []
    assert other.get(f"/api/conversations/{body['conversation_id']}").status_code == 404


def test_streaming_chat():
    c = new_client(); register(c, "stream@x.com")
    upload(c, NETWORKS, "net.pdf")
    r = c.post("/api/chat/stream", json={"question": "What do routers do with packets?"})
    assert r.status_code == 200
    assert "event: meta" in r.text and "event: delta" in r.text and "event: done" in r.text
    assert len(c.get("/api/conversations").json()) == 1


def test_originality_checks():
    a = new_client(); register(a, "orig-a@x.com")
    assert upload(a, NETWORKS, "networks.pdf").json()["report"]["verdict"] == "unique"

    # same file again -> blocked as exact duplicate
    dup = upload(a, NETWORKS, "networks-copy.pdf")
    assert dup.status_code == 409 and dup.json()["detail"]["code"] == "duplicate_file"

    # lightly edited copy -> flagged as similar, can be overridden
    edited = NETWORKS.replace("fair and reliable", "fair and dependable")
    sim = upload(a, edited, "networks-v2.pdf")
    assert sim.status_code == 409
    rep = sim.json()["detail"]["report"]
    assert rep["verdict"] == "high" and rep["matches"][0]["filename"] == "networks.pdf"
    assert rep["matches"][0]["passages"]
    assert upload(a, edited, "networks-v2.pdf", force=True).status_code == 200

    # unrelated document -> fine
    assert upload(a, COOKING, "bread.pdf").json()["report"]["verdict"] == "unique"

    # another user copying it is caught, but sees no filename or text
    b = new_client(); register(b, "orig-b@x.com")
    sim_b = upload(b, NETWORKS, "mine.pdf")
    assert sim_b.status_code == 409
    m = sim_b.json()["detail"]["report"]["matches"][0]
    assert m["filename"] == "Another user's document" and "passages" not in m and "document_id" not in m

    # on-demand report for an existing document
    doc_id = a.get("/api/documents").json()[-1]["id"]
    assert a.get(f"/api/documents/{doc_id}/similarity").status_code == 200
    assert b.get(f"/api/documents/{doc_id}/similarity").status_code == 404


def test_rate_limit():
    ratelimit.ENABLED = True
    ratelimit._hits.clear()
    try:
        c = new_client()
        codes = [c.post("/api/login", json={"email": "x@x.com", "password": "password1"}).status_code for _ in range(12)]
        assert codes[:10] == [401] * 10 and codes[10] == 429
    finally:
        ratelimit.ENABLED = False
        ratelimit._hits.clear()
