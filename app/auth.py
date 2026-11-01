import hashlib
import hmac
import os
import secrets

from fastapi import Cookie, HTTPException, Response

from . import db

COOKIE = "session"
MAX_AGE = 7 * 24 * 3600
SECURE = os.getenv("COOKIE_SECURE", "0") == "1"  # set to 1 behind HTTPS


def hash_password(password: str, salt: str | None = None):
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 200_000).hex()
    return salt, digest


def verify_password(password: str, salt: str, expected: str) -> bool:
    return hmac.compare_digest(hash_password(password, salt)[1], expected)


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()  # only the hash is stored server-side


def create_session(user_id: int, response: Response) -> None:
    token = secrets.token_urlsafe(32)
    with db.conn() as c:
        c.execute("DELETE FROM sessions WHERE created_at < datetime('now','-7 days')")
        c.execute("INSERT INTO sessions(token, user_id) VALUES(?,?)", (_digest(token), user_id))
    response.set_cookie(COOKIE, token, max_age=MAX_AGE, httponly=True, samesite="lax", secure=SECURE, path="/")


def end_session(token: str | None, response: Response) -> None:
    if token:
        with db.conn() as c:
            c.execute("DELETE FROM sessions WHERE token=?", (_digest(token),))
    response.delete_cookie(COOKIE, path="/")


def current_user(session: str | None = Cookie(None)) -> dict:
    if not session:
        raise HTTPException(401, "Not signed in")
    with db.conn() as c:
        row = c.execute(
            "SELECT u.id, u.name, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
            "WHERE s.token=? AND s.created_at > datetime('now','-7 days')",
            (_digest(session),),
        ).fetchone()
    if not row:
        raise HTTPException(401, "Session expired, please sign in again")
    return dict(row)
