# app.py — Chatbot UniSalute IPZS 2026 (RAG "serio" stile consulente UILCOM)
# - Domande naturali (oculistica, odontoiatria, rimborsi, procedure)
# - NON mostra testo guida
# - Mostra SOLO riferimenti pagina (Rif.: Pag. X–Y)
# - Guardrail: se retrieval debole => "Non ho trovato..."
#
# Requisiti (requirements.txt):
# streamlit
# pymupdf
# openai
# rank-bm25

import os
import re
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import fitz  # PyMuPDF
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -------------------------
# Config
# -------------------------
PDF_PATH = "Unis2026.pdf"

STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","si","sì","no",
    "del","dello","della","dei","degli","delle",
    "al","allo","alla","ai","agli","alle",
    "nel","nello","nella","nei","negli","nelle",
    "puo","può","posso","fare","faccio","gratis","gratuita","gratuito","costo","costare",
    "una","uno","un"
}

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def tokenize(text: str) -> List[str]:
    parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", (text or "").lower())
    toks = [p for p in parts if len(p) >= 3 and p not in STOPWORDS_IT]
    # dedup mantenendo ordine
    seen = set()
    out = []
    for t in toks:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def pages_ref_string(pages: List[int]) -> str:
    """Compatta (Pag. 3, 4, 5, 9, 10) -> Pag. 3–5, 9–10"""
    if not pages:
        return ""
    pages = sorted(set(pages))
    ranges = []
    start = prev = pages[0]
    for p in pages[1:]:
        if p == prev + 1:
            prev = p
        else:
            ranges.append((start, prev))
            start = prev = p
    ranges.append((start, prev))
    chunks = []
    for a, b in ranges:
        chunks.append(f"Pag. {a}–{b}" if a != b else f"Pag. {a}")
    return ", ".join(chunks)

def get_openai_key() -> str:
    key = ""
    try:
        key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        key = ""
    return key or os.getenv("OPENAI_API_KEY", "")


# -------------------------
# PDF -> chunks (con pagina)
# -------------------------
@dataclass
class Chunk:
    page: int
    text: str
    tokens: List[str]

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 180) -> List[str]:
    """Chunk semplice a caratteri con overlap (robusto su PDF)."""
    text = normalize_spaces(text)
    if not text:
        return []
    out = []
    i = 0
    n = len(text)
    while i < n:
        j = min(n, i + chunk_size)
        out.append(text[i:j])
        if j == n:
            break
        i = max(0, j - overlap)
    return out

@st.cache_resource
def build_bm25_index(pdf_path: str, chunk_size: int, overlap: int) -> Tuple[List[Chunk], BM25Okapi]:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF non trovato: {pdf_path}")

    doc = fitz.open(pdf_path)
    chunks: List[Chunk] = []

    for i in range(doc.page_count):
        page_num = i + 1
        raw = doc.load_page(i).get_text("text") or ""
        raw = normalize_spaces(raw)
        for part in chunk_text(raw, chunk_size=chunk_size, overlap=overlap):
            toks = tokenize(part)
            if toks:
                chunks.append(Chunk(page=page_num, text=part, tokens=toks))

    if not chunks:
        raise ValueError("Nessun testo indicizzabile nel PDF.")

    bm25 = BM25Okapi([c.tokens for c in chunks])
    return chunks, bm25

def retrieve_chunks(chunks: List[Chunk], bm25: BM25Okapi, question: str, top_k: int) -> List[Tuple[float, Chunk]]:
    q_tokens = tokenize(question)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    out: List[Tuple[float, Chunk]] = []
    for idx in ranked_idx[: top_k * 3]:  # prendo più candidati, poi dedup
        s = float(scores[idx])
        if s <= 0:
            continue
        out.append((s, chunks[idx]))

    # dedup per pagina+testo (evita ripetizioni)
    seen = set()
    deduped = []
    for s, c in out:
        key = (c.page, c.text[:80])
        if key in seen:
            continue
        seen.add(key)
        deduped.append((s, c))

    return deduped[:top_k]


# -------------------------
# LLM Answer (solo risposta, no testo guida)
# -------------------------
def llm_answer_consulente_uilcom(question: str, retrieved: List[Tuple[float, Chunk]]) -> str:
    """
    Il modello riceve il CONTENUTO GUIDA ma:
    - NON deve citarlo né copiarlo
    - deve dare risposta pratica
    - se non basta: non trovato
    """
    api_key = get_openai_key()
    if not api_key or OpenAI is None:
        # Senza chiave: non possiamo generare risposta "consulente"
        return "Per avere la risposta in stile consulente (senza mostrare testo), inserisci OPENAI_API_KEY nei Secrets di Streamlit."

    # Contesto (interno) — non verrà mostrato a video
    context_blocks = []
    for _, c in retrieved:
        context_blocks.append(f"[Pag. {c.page}] {c.text}")
    context = "\n\n".join(context_blocks)

    client = OpenAI(api_key=api_key)

    system = (
        "Sei un consulente sindacale UILCOM. Rispondi ad iscritti su Piano sanitario integrativo (UniSalute). "
        "Devi essere chiaro, pratico, preciso.\n\n"
        "REGOLE HARD:\n"
        "1) Usa SOLO il contenuto fornito come 'CONTENUTO GUIDA'.\n"
        "2) NON riportare MAI frasi/estratti della guida, né tra virgolette né in forma di copia-incolla.\n"
        "3) NON inventare: massimali, franchigie, scoperti, limiti, condizioni, procedure o definizioni non presenti.\n"
        "4) Se il contenuto non permette risposta certa, rispondi ESATTAMENTE: "
        "'Non ho trovato nella guida UniSalute una risposta certa a questa domanda.'\n"
        "5) Se l’utente dice 'gratis/gratuita/gratuito', interpreta come: 'prestazione coperta dal piano? con quali modalità/condizioni?'\n"
        "6) Non citare pagine: le pagine le aggiunge l’app.\n"
    )

    user = f"""
DOMANDA UTENTE:
{question}

CONTENUTO GUIDA (UNICA FONTE):
{context}

FORMATO RISPOSTA (OBBLIGATORIO):
- ESITO: (Coperta / Non citata / Serve verifica) scegli una sola voce.
- RISPOSTA: 2–6 righe, diretta.
- COME FARE: 3–7 punti operativi (solo se ricavabili).
- ATTENZIONI: 0–5 punti (limiti, condizioni, esclusioni) SOLO se nel testo.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content.strip()


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Chatbot UniSalute IPZS 2026", layout="wide")
st.title("🤖 Chatbot UniSalute IPZS 2026")
st.caption("Domande naturali, risposta stile consulente UILCOM. Nessun testo guida mostrato. Solo riferimenti pagina.")

with st.sidebar:
    st.header("⚙️ Impostazioni")
    chunk_size = st.slider("Dimensione blocchi (chunk)", 700, 2200, 1200, step=100)
    overlap = st.slider("Sovrapposizione (overlap)", 0, 400, 180, step=20)
    top_k = st.slider("Pagine/blocchi usati (top-k)", 3, 12, 7, step=1)
    strict = st.toggle("Modalità rigorosa (consigliata)", value=True)
    min_hits = st.slider("Soglia minima (quanti blocchi devono combaciare)", 1, 5, 1)
    st.divider()
    st.caption("Per risposte complete serve OPENAI_API_KEY nei Secrets di Streamlit.")

# indicizzazione
try:
    chunks, bm25 = build_bm25_index(PDF_PATH, chunk_size=chunk_size, overlap=overlap)
except Exception as e:
    st.error(f"Errore indicizzazione PDF: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Ciao! Chiedimi pure: “Visita oculistica: è coperta? come fare?” oppure “Impianto dentale: è previsto?”."}
    ]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Scrivi una domanda (es: 'Visita oculistica gratuita come fare?')")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = retrieve_chunks(chunks, bm25, question, top_k=top_k)

    # calcolo pagine dai retrieved
    pages = [c.page for _, c in retrieved]
    pages_str = pages_ref_string(pages)

    # Guardrail: se retrieval troppo debole
    if strict:
        if len(retrieved) < min_hits:
            answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
            if pages_str:
                answer += f"\n\n**Rif.:** {pages_str}"
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            answer = llm_answer_consulente_uilcom(question, retrieved)
            # aggiungo sempre pagine (solo pagine)
            if pages_str:
                answer = f"{answer}\n\n**Rif.:** {pages_str}"
            with st.chat_message("assistant"):
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
    else:
        # modalità non rigorosa: prova comunque
        answer = llm_answer_consulente_uilcom(question, retrieved)
        if pages_str:
            answer = f"{answer}\n\n**Rif.:** {pages_str}"
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
