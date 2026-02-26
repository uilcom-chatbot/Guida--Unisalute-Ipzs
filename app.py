import os
import re
import fitz  # pymupdf
import streamlit as st
from rank_bm25 import BM25Okapi

# OpenAI (opzionale)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

st.set_page_config(page_title="Chatbot UniSalute IPZS 2026", layout="wide")

st.title("🤖 Chatbot UniSalute IPZS 2026")
st.caption("Fai domande in linguaggio naturale. La risposta usa SOLO la guida e riporta le pagine. Se non trova, lo dice.")

PDF_PATH = "Unis2026.pdf"

# -------------------------
# Helpers
# -------------------------
STOPWORDS = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","si","sì","no","del","dello","della",
    "dei","degli","delle","al","allo","alla","ai","agli","alle","nel","nello","nella","nei","negli","nelle",
    "puo","può","fare","faccio","farsi","gratis","gratuita","gratuito"
}

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def tokenize(text: str) -> list[str]:
    parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", (text or "").lower())
    toks = [p for p in parts if len(p) >= 3 and p not in STOPWORDS]
    return toks

def extract_pages(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(doc.page_count):
        t = normalize_spaces(doc.load_page(i).get_text("text") or "")
        if t:
            pages.append({"page": i + 1, "text": t})
    return pages

@st.cache_resource
def build_index(pdf_path: str):
    pages = extract_pages(pdf_path)
    corpus_tokens = [tokenize(p["text"]) for p in pages]
    bm25 = BM25Okapi(corpus_tokens)
    return pages, bm25, corpus_tokens

def retrieve(pages, bm25, query: str, top_k: int = 6):
    q_tokens = tokenize(query)
    if not q_tokens:
        return []
    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    results = []
    for idx in ranked[:top_k]:
        if scores[idx] <= 0:
            continue
        results.append({
            "page": pages[idx]["page"],
            "text": pages[idx]["text"],
            "score": float(scores[idx]),
        })
    return results

def build_context(chunks, max_chars: int = 6000):
    # unisce estratti con tag pagina
    out = []
    total = 0
    for c in chunks:
        block = f"[Pag. {c['page']}] {c['text']}"
        if total + len(block) > max_chars:
            break
        out.append(block)
        total += len(block)
    return "\n\n".join(out)

def answer_with_llm(question: str, context: str) -> str:
    api_key = st.secrets.get("OPENAI_API_KEY", "") if hasattr(st, "secrets") else ""
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    if not api_key or OpenAI is None:
        return ""  # modalità senza LLM

    client = OpenAI(api_key=api_key)

    system = (
        "Sei un assistente che risponde SOLO usando il testo fornito come CONTENUTO GUIDA. "
        "Se l'informazione non è nel contenuto guida, devi rispondere: "
        "'Non ho trovato nella guida UniSalute una risposta certa a questa domanda.' "
        "Non devi inventare massimali, percentuali o condizioni. "
        "Quando possibile, spiega 'come fare' (procedura) in modo pratico e indica le pagine citate."
    )

    user = f"""
DOMANDA UTENTE:
{question}

CONTENUTO GUIDA (usa solo questo):
{context}

ISTRUZIONI RISPOSTA:
- Rispondi in italiano, chiaro e pratico.
- Se la domanda usa 'gratuita/gratis', interpreta come: 'coperta dal piano? con quali condizioni/rimborso?'
- Cita sempre le pagine: es. (Pag. 12, 13).
- Se non è nel testo: usa la frase di non trovato.
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
with st.sidebar:
    st.header("⚙️ Impostazioni")
    top_k = st.slider("Pagine candidate (top-k)", 3, 12, 6)
    strict = st.toggle("Modalità rigorosa (consigliata)", value=True)
    show_sources = st.toggle("Mostra sempre le fonti", value=True)
    st.caption("Se metti OPENAI_API_KEY nei Secrets, la chatbot scrive una risposta completa. Altrimenti mostra estratti trovati.")

# Build index
try:
    pages, bm25, _ = build_index(PDF_PATH)
except Exception as e:
    st.error("Errore nel leggere il PDF. Controlla che 'Unis2026.pdf' sia nel repo.")
    st.stop()

if "chat" not in st.session_state:
    st.session_state.chat = []

# Chat history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

question = st.chat_input("Scrivi una domanda (es. 'Visita oculistica: come fare per ottenere il rimborso?')")

if question:
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    chunks = retrieve(pages, bm25, question, top_k=top_k)

    # Guardrail: se non troviamo nulla, stop
    if strict and not chunks:
        answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
        with st.chat_message("assistant"):
            st.markdown(answer)
        st.session_state.chat.append({"role": "assistant", "content": answer})
    else:
        context = build_context(chunks)

        answer = answer_with_llm(question, context)

        # Se non c'è chiave o LLM non disponibile -> fallback: estratti
        if not answer:
            if not chunks:
                answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
            else:
                answer = "Ho trovato questi estratti nella guida (controlla le pagine):"

        with st.chat_message("assistant"):
            st.markdown(answer)

            if show_sources and chunks:
                st.divider()
                st.subheader("📌 Fonti dalla guida")
                # mostra estratti più brevi per leggibilità
                for c in chunks:
                    snippet = c["text"][:900] + ("…" if len(c["text"]) > 900 else "")
                    st.markdown(f"**Pag. {c['page']}** — {snippet}")

        st.session_state.chat.append({"role": "assistant", "content": answer})
