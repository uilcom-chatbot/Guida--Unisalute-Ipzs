import re
import fitz  # PyMuPDF
import streamlit as st

st.set_page_config(page_title="UniSalute IPZS 2026", layout="wide")

st.title("🔎 UniSalute IPZS 2026")
st.caption("Scrivi parole o frasi (es: 'visita oculistica rimborso quanto paga'). L'app mostra SOLO estratti del PDF con pagina.")

PDF_PATH = "Unis2026.pdf"

# Stopwords base (puoi aggiungerne quante vuoi)
STOPWORDS = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","sì","no","del","dello","della",
    "dei","degli","delle","al","allo","alla","ai","agli","alle","nel","nello","nella","nei","negli","nelle"
}

with st.sidebar:
    st.header("⚙️ Opzioni")
    mode = st.radio("Criterio", ["Tutte le parole (più preciso)", "Almeno una parola (più largo)"], index=0)
    max_hits = st.slider("Risultati massimi", 1, 30, 10)
    context_chars = st.slider("Lunghezza estratto", 160, 1200, 380, step=20)
    st.caption("Suggerimento: usa parole singole (es. 'oculistica', 'rimborso', 'ticket').")

query = st.text_input("Parola chiave o frase", placeholder="Esempio: visita oculistica rimborso quanto paga")
do_search = st.button("Cerca", type="primary", use_container_width=True)

def normalize_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def tokenize(q: str) -> list[str]:
    # prende solo lettere/numeri, separa, minuscolo
    parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", q.lower())
    # filtra stopwords e token troppo corti
    tokens = [p for p in parts if p not in STOPWORDS and len(p) >= 3]
    # dedup mantenendo ordine
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            out.append(t)
            seen.add(t)
    return out

def highlight_many(text: str, words: list[str]) -> str:
    # Evidenzia tutte le parole (case-insensitive) senza rompere HTML
    if not words:
        return text
    # ordina per lunghezza decrescente per evitare sovrapposizioni
    words_sorted = sorted(words, key=len, reverse=True)
    pattern = re.compile(r"(" + "|".join(map(re.escape, words_sorted)) + r")", re.IGNORECASE)
    return pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", text)

def best_snippet(text: str, words: list[str], ctx: int) -> str:
    """
    Trova un punto 'buono' dove compaiono parole (la prima occorrenza migliore) e crea snippet.
    """
    if not words:
        return ""
    # trova tutte le posizioni delle parole
    positions = []
    for w in words:
        m = re.search(re.escape(w), text, re.IGNORECASE)
        if m:
            positions.append(m.start())
    if not positions:
        return ""
    pos = min(positions)
    start = max(0, pos - ctx // 2)
    end = min(len(text), pos + ctx // 2)
    return normalize_spaces(text[start:end])

if do_search:
    q = (query or "").strip()
    tokens = tokenize(q)

    if len(tokens) == 0:
        st.warning("Scrivi almeno una parola utile (evita solo 'come', 'quanto', 'il', ecc.).")
        st.stop()

    # apri PDF
    doc = fitz.open(PDF_PATH)

    results = []  # (score, page_num, snippet, matched_tokens)

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = normalize_spaces(page.get_text("text") or "")

        matched = [t for t in tokens if re.search(re.escape(t), text, re.IGNORECASE)]
        if not matched:
            continue

        if mode.startswith("Tutte") and len(matched) < len(tokens):
            continue

        score = len(matched)  # quante parole ha trovato in quella pagina
        snippet = best_snippet(text, matched, context_chars)
        if snippet:
            results.append((score, i + 1, snippet, matched))

    if not results:
        st.error("❌ Non presente nella guida (nessuna corrispondenza trovata con i criteri scelti).")
    else:
        # ordina: prima più match, poi pagina
        results.sort(key=lambda x: (-x[0], x[1]))

        st.success(f"✅ Trovate {len(results)} pagine pertinenti. Mostro fino a {max_hits}.")
        for n, (score, page_num, snippet, matched) in enumerate(results[:max_hits], start=1):
            st.write(f"**Risultato {n} — Pagina {page_num} — match {score}/{len(tokens)}**")
            st.markdown(highlight_many(snippet, matched), unsafe_allow_html=True)
            st.caption("Parole trovate: " + ", ".join(matched))
            st.divider()
