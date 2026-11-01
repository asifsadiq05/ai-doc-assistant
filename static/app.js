const $ = (s) => document.querySelector(s);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
let convId = null;
let registering = false;

class ApiError extends Error {
  constructor(message, status, detail) { super(message); this.status = status; this.detail = detail; }
}

async function api(path, { json, ...opts } = {}) {
  const headers = { "X-Requested-With": "docsmith", ...(opts.headers || {}) };
  if (json) { headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(json); }
  const r = await fetch("/api" + path, { ...opts, headers });
  const data = await r.json().catch(() => ({}));
  const d = data.detail;
  if (r.status === 401 && !/^\/(login|register|logout|me)$/.test(path)) signOut();
  if (!r.ok) {
    const msg = typeof d === "string" ? d : Array.isArray(d) ? "Check the fields and try again" : d?.message || "Something went wrong";
    throw new ApiError(msg, r.status, d);
  }
  return data;
}

/* ---------- auth (session lives in an httpOnly cookie) ---------- */
function setMode() {
  $("#a-name").hidden = !registering;
  $("#a-name").required = registering;
  $("#a-submit").textContent = registering ? "Create account" : "Sign in";
  $("#a-toggle").textContent = registering ? "I already have an account" : "Create an account";
  $("#a-pass").autocomplete = registering ? "new-password" : "current-password";
  $("#a-err").textContent = "";
}
$("#a-toggle").onclick = () => { registering = !registering; setMode(); };
$("#auth-form").onsubmit = async (e) => {
  e.preventDefault();
  try {
    const body = { email: $("#a-email").value, password: $("#a-pass").value };
    if (registering) body.name = $("#a-name").value;
    await api(registering ? "/register" : "/login", { method: "POST", json: body });
    $("#a-pass").value = "";
    start();
  } catch (err) { $("#a-err").textContent = err.message; }
};
function signOut() {
  convId = null;
  $("#app").hidden = true; $("#auth").hidden = false; setMode();
}
$("#logout").onclick = async () => { try { await api("/logout", { method: "POST" }); } catch {} signOut(); };

/* ---------- originality report ---------- */
const VERDICT = {
  high: ["High overlap found", "Most of this document already exists in the database."],
  partial: ["Partial overlap", "Some passages match existing documents."],
  topic: ["Same topic, different wording", "The subject matches existing documents, but the text is not copied."],
  unique: ["No significant overlap", "This document looks original compared with the database."],
};
const pct = (x) => Math.round(x * 100) + "%";

function showReport(r, { title = "", confirm } = {}) {
  const [head, sub] = VERDICT[r.verdict];
  let html = `<h2 id="r-title" class="${r.verdict === "high" ? "v-high" : ""}">${esc(head)}</h2>
    <p class="sub">${esc(title)}${title ? " · " : ""}${esc(sub)}</p>
    <div class="meter ${r.verdict === "high" ? "bad" : ""}"><div style="width:${r.originality}%"></div></div>
    <p class="hint">Originality ${r.originality}% · compared with ${r.checked_against} document${r.checked_against === 1 ? "" : "s"}</p>`;
  r.matches.forEach((m) => {
    html += `<div class="match"><b>${esc(m.filename)}</b>
      <div class="bar"><div style="width:${Math.round(m.overlap * 100)}%"></div></div>
      <span class="hint">${pct(m.overlap)} of this text appears there · topic similarity ${pct(m.topic)}</span>
      ${(m.passages || []).map((p) => `<q>p.${p.page}: ${esc(p.snippet)}…</q>`).join("")}</div>`;
  });
  if (!r.matches.length) html += '<p class="hint">No similar documents found.</p>';
  $("#report-body").innerHTML = html;
  const ok = $("#r-confirm");
  ok.hidden = !confirm;
  ok.onclick = () => { $("#report").close(); confirm(); };
  $("#report").showModal();
}
$("#r-close").onclick = () => $("#report").close();

/* ---------- documents ---------- */
async function loadDocs() {
  const docs = await api("/documents");
  const ul = $("#docs"); ul.innerHTML = "";
  docs.forEach((d) => {
    const li = document.createElement("li");
    li.innerHTML = `<input type="checkbox" value="${d.id}" aria-label="Use ${esc(d.filename)}" style="width:auto">
      <span title="${esc(d.filename)}">${esc(d.filename)}</span><small>${d.pages}p</small>
      <button class="chk" title="Check originality">Check</button><button aria-label="Delete ${esc(d.filename)}">✕</button>`;
    const [chk, del] = li.querySelectorAll("button");
    chk.onclick = async () => {
      try { showReport(await api(`/documents/${d.id}/similarity`), { title: d.filename }); }
      catch (err) { $("#up-status").textContent = err.message; }
    };
    del.onclick = async () => {
      if (confirm(`Delete "${d.filename}"?`)) { await api("/documents/" + d.id, { method: "DELETE" }); loadDocs(); }
    };
    ul.appendChild(li);
  });
  if (!docs.length) ul.innerHTML = '<li class="hint">No documents yet.</li>';
}

async function uploadFile(f, force = false) {
  const st = $("#up-status"); st.textContent = "Reading and checking " + f.name + "…";
  const fd = new FormData(); fd.append("file", f);
  try {
    const d = await api("/documents" + (force ? "?force=true" : ""), { method: "POST", body: fd });
    st.textContent = `Indexed ${d.pages} pages into ${d.chunks} passages.`;
    loadDocs();
    if (d.report.verdict !== "unique") showReport(d.report, { title: f.name });
  } catch (err) {
    if (err.detail?.code === "similar_content") {
      st.textContent = "Upload paused: this content already exists.";
      showReport(err.detail.report, { title: f.name, confirm: () => uploadFile(f, true) });
    } else st.textContent = err.message;
  }
}
$("#file").onchange = (e) => { const f = e.target.files[0]; if (f) uploadFile(f); e.target.value = ""; };

/* ---------- chats ---------- */
async function loadConvs() {
  const cs = await api("/conversations");
  const ul = $("#convs"); ul.innerHTML = "";
  cs.forEach((c) => {
    const li = document.createElement("li");
    if (c.id === convId) li.className = "active";
    li.innerHTML = `<span>${esc(c.title)}</span><button aria-label="Delete chat">✕</button>`;
    li.querySelector("span").onclick = () => openConv(c.id);
    li.querySelector("button").onclick = async () => {
      await api("/conversations/" + c.id, { method: "DELETE" });
      if (convId === c.id) newChat(); else loadConvs();
    };
    ul.appendChild(li);
  });
}
async function openConv(id) {
  convId = id;
  const msgs = await api("/conversations/" + id);
  $("#messages").innerHTML = "";
  msgs.forEach((m) => addMsg(m.role, m.content, m.sources));
  loadConvs();
}
function newChat() {
  convId = null;
  $("#messages").innerHTML = '<p class="empty">Upload a PDF, then ask a question. Each answer lists the pages it used.</p>';
  loadConvs();
}
$("#new-chat").onclick = newChat;

function renderSources(div, sources) {
  if (!sources.length) return;
  const wrap = document.createElement("div"); wrap.className = "srcs";
  const snip = document.createElement("div"); snip.className = "snip"; snip.hidden = true;
  sources.forEach((s, i) => {
    const b = document.createElement("button"); b.className = "src";
    b.textContent = `[${i + 1}] ${s.filename} · p.${s.page}`;
    b.onclick = () => { snip.hidden = false; snip.textContent = "…" + s.snippet + "…"; };
    wrap.appendChild(b);
  });
  div.append(wrap, snip);
}
function addMsg(role, text, sources = []) {
  const box = $("#messages");
  box.querySelector(".empty")?.remove();
  const div = document.createElement("div");
  div.className = "msg " + role;
  const body = document.createElement("span"); body.textContent = text; div.appendChild(body);
  renderSources(div, sources);
  box.appendChild(div); box.scrollTop = box.scrollHeight;
  return div;
}

/* streaming answers over Server-Sent Events */
async function streamChat(q, ids, bubble) {
  const r = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json", "X-Requested-With": "docsmith" },
    body: JSON.stringify({ question: q, conversation_id: convId, document_ids: ids.length ? ids : null }),
  });
  if (r.status === 401) signOut();
  if (!r.ok) { const d = await r.json().catch(() => ({})); throw new Error(typeof d.detail === "string" ? d.detail : "Something went wrong"); }
  const reader = r.body.getReader(), dec = new TextDecoder();
  const out = bubble.firstChild; let buf = "", text = "", sources = [];
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    let i;
    while ((i = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, i); buf = buf.slice(i + 2);
      const ev = /^event: (.+)$/m.exec(block)?.[1];
      const data = JSON.parse(/^data: (.+)$/m.exec(block)?.[1] || "{}");
      if (ev === "meta") sources = data.sources;
      else if (ev === "delta") { text += data.text; out.textContent = text; bubble.classList.remove("thinking"); $("#messages").scrollTop = 1e9; }
      else if (ev === "error") throw new Error(data.message);
      else if (ev === "done") convId = data.conversation_id;
    }
  }
  renderSources(bubble, sources);
}

$("#chat-form").onsubmit = async (e) => {
  e.preventDefault();
  const q = $("#q").value.trim(); if (!q) return;
  $("#q").value = ""; $("#send").disabled = true;
  addMsg("user", q);
  const bubble = addMsg("assistant", "Searching your documents…"); bubble.classList.add("thinking");
  const ids = [...document.querySelectorAll("#docs input:checked")].map((c) => +c.value);
  try { await streamChat(q, ids, bubble); loadConvs(); }
  catch (err) { bubble.firstChild.textContent = err.message; bubble.classList.remove("thinking"); }
  $("#send").disabled = false; $("#q").focus();
};
$("#q").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
});

function start() { $("#auth").hidden = true; $("#app").hidden = false; newChat(); loadDocs(); }
api("/me").then(start).catch(() => { $("#auth").hidden = false; setMode(); });
