# app.py — Chatbot UniSalute IPZS 2026 v2
# Miglioramenti:
# 1. Memoria conversazione (contesto domande precedenti)
# 2. Ricerca ibrida BM25 + TF-IDF
# 3. Reranking risultati
# 4. Chunk per sezioni logiche della guida

import os
import re
import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_PATH = "chunks_data.json"

STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","si","sì","no",
    "del","dello","della","dei","degli","delle","al","allo","alla","ai","agli","alle",
    "nel","nello","nella","nei","negli","nelle","puo","può","posso","fare","faccio",
    "gratis","gratuita","gratuito","costo","costare","anche","sono","essere","avere",
    "questo","questa","questi","queste","quello","quella","quelli","quelle",
}

SYNONYMS = {
    "dentista":       ["odontoiatriche","odontoiatrico","odontoiatria","dentale","cure"],
    "denti":          ["odontoiatriche","dentale","odontoiatria","elemento","carie"],
    "dente":          ["odontoiatriche","dentale","impianto","elemento","fixture"],
    "impianto":       ["implantologia","fixture","impianto","posizionamento"],
    "impianti":       ["implantologia","fixture","impianto","posizionamento"],
    "protesi":        ["protesi","ortodonzia","protesica","corona","fixture","ortopediche"],
    "apparecchio":    ["ortodonzia","protesi","ortodontico"],
    "pulizia":        ["igiene","ablazione","tartaro","ultrasuoni","orale","odontoiatriche"],
    "igiene":         ["igiene","ablazione","tartaro","orale","sedute"],
    "tartaro":        ["ablazione","tartaro","igiene","ultrasuoni","orale"],
    "otturazione":    ["conservativa","terapie","carie","odontoiatrica"],
    "carie":          ["conservativa","carie","odontoiatrica","cure"],
    "estrazione":     ["estrazione","chirurgico","odontoiatrica","avulsione"],
    "occhiali":       ["lenti","occhiali","correttive","montatura","visus","oculista"],
    "lenti":          ["lenti","occhiali","correttive","montatura","contatto","visus"],
    "vista":          ["visus","oculista","visiva","oculistica","lenti","occhiali"],
    "oculista":       ["oculista","oculistica","visus","visiva","lenti"],
    "miopia":         ["visus","oculista","lenti","correttive","occhiali"],
    "fisioterapia":   ["fisioterapici","riabilitativi","fisioterapia","riabilitazione","trattamenti","malattia"],
    "fisio":          ["fisioterapici","riabilitativi","fisioterapia","riabilitazione"],
    "riabilitazione": ["riabilitativi","fisioterapici","riabilitazione","trattamenti"],
    "massaggio":      ["fisioterapici","riabilitativi","trattamenti","paramedico"],
    "ospedale":       ["ricovero","ospedaliero","istituto","cura","degenza"],
    "operazione":     ["chirurgico","intervento","chirurgia","operatorio"],
    "intervento":     ["chirurgico","intervento","operatorio","chirurgia"],
    "parto":          ["parto","cesareo","neonati","gravidanza"],
    "visita":         ["specialistiche","visita","specialista","prescrizione"],
    "esame":          ["diagnostici","accertamenti","esami","prescrizione","diagnostica"],
    "esami":          ["diagnostici","accertamenti","esami","prescrizione","diagnostica"],
    "analisi":        ["diagnostici","accertamenti","esami","laboratorio","cliniche"],
    "allergia":       ["accertamenti","diagnostici","specialistiche","prescrizione","patologia","allergologica"],
    "allergologia":   ["accertamenti","diagnostici","specialistiche","prescrizione"],
    "sangue":         ["ematologici","diagnostici","accertamenti","esami","laboratorio"],
    "radiografia":    ["radiologica","diagnostica","accertamenti","rx","diagnostici"],
    "ecografia":      ["ecografia","ecocolordoppler","diagnostici","accertamenti"],
    "risonanza":      ["risonanza","magnetica","diagnostici","alta","specializzazione"],
    "tac":            ["tac","diagnostici","alta","specializzazione","accertamenti"],
    "farmaci":        ["medicinali","farmaci","farmacologica","prescritti","curante"],
    "medicine":       ["medicinali","farmaci","prescritti","curante"],
    "rimborso":       ["rimborso","rimborsate","rimborsuale","rimborsi","richiesta"],
    "rimborsare":     ["rimborso","rimborsate","rimborsuale","rimborsi"],
    "pagamento":      ["rimborso","liquidate","liquidazione","massimale","franchigia"],
    "convenzionato":  ["convenzionata","convenzionato","strutture","rete"],
    "anziano":        ["autosufficienza","non autosufficienza","ltc","assistenza"],
    "badante":        ["assistenza","socio","assistenziali","badanti","domiciliare"],
    "psicologia":     ["psichiatrica","psichici","mentali"],
    "termale":        ["termali","terme","termale","cure"],
    "udito":          ["acustiche","udito","protesi","otoemissioni"],
    "ticket":         ["ticket","ssn","nazionale","sanitari","rimborso"],
    "estero":         ["estero","internazionale","rimpatrio","validita","territoriale"],
    "secondo parere": ["second","opinion","parere","specialista","diagnosi"],
    "non autosufficienza": ["autosufficienza","ltc","assistenza","permanente"],
    "massimale":      ["massimale","limite","annuo","nucleo","familiare"],
    "franchigia":     ["franchigia","scoperto","minimo","indennizzabile"],
    "familiare":      ["familiare","nucleo","coniuge","figli","famiglia"],
    "figlio":         ["figli","familiare","nucleo","coniuge","minorenne"],
    "infortunio":     ["infortunio","infortuni","trauma","pronto","soccorso"],
    "day hospital":   ["day","hospital","ambulatoriale","chirurgico"],
    "ricovero":       ["ricovero","degenza","ospedaliero","chirurgico","medico"],
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def expand_query(text: str) -> str:
    lower = text.lower()
    extra = []
    for keyword, terms in SYNONYMS.items():
        if keyword in lower:
            extra.extend(terms)
    return text + (" " + " ".join(extra) if extra else "")

def tokenize(text: str) -> List[str]:
    parts = re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿ0-9]+", (text or "").lower())
    seen = set()
    out = []
    for p in parts:
        if len(p) >= 3 and p not in STOPWORDS_IT and p not in seen:
            out.append(p)
            seen.add(p)
    return out

def pages_ref_string(pages: List[int]) -> str:
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
    return ", ".join(f"Pag. {a}–{b}" if a != b else f"Pag. {a}" for a, b in ranges)

def get_api_key() -> str:
    key = ""
    if hasattr(st, "secrets"):
        try:
            key = st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    if not key:
        key = os.getenv("OPENAI_API_KEY", "")
    return (key or "").strip()

def check_password() -> bool:
    correct_pw = ""
    if hasattr(st, "secrets"):
        try:
            correct_pw = st.secrets.get("APP_PASSWORD", "")
        except Exception:
            pass
    if not correct_pw:
        correct_pw = os.getenv("APP_PASSWORD", "Uni79")

    if st.session_state.get("authenticated"):
        return True

    st.markdown("""
    <div style='text-align:center; padding: 40px 0 10px 0;'>
        <span style='font-size:3rem'>🏥</span>
        <h2 style='margin:8px 0 4px 0'>Chatbot UniSalute IPZS 2026</h2>
        <p style='color:gray'>Consulente Sindacale UILCOM — Accesso riservato</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        pwd = st.text_input("Password", type="password", placeholder="Inserisci la password")
        if st.button("Accedi →", use_container_width=True):
            if pwd == correct_pw:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Password errata. Riprova.")
    return False


# ── #4 Chunk Index (sezioni logiche + overlap) ────────────────────────────────
@dataclass
class Chunk:
    page: int
    pages: List[int]
    text: str
    tokens: List[str]

@st.cache_resource
def build_index(chunks_path: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    chunks = []
    for item in raw:
        toks = tokenize(item["text"])
        if toks:
            chunks.append(Chunk(
                page=item["page"],
                pages=item.get("pages", [item["page"]]),
                text=item["text"],
                tokens=toks
            ))

    # BM25
    bm25 = BM25Okapi([c.tokens for c in chunks])

    # #2 TF-IDF index
    N = len(chunks)
    df: Dict[str, int] = {}
    for c in chunks:
        for t in set(c.tokens):
            df[t] = df.get(t, 0) + 1
    idf = {t: math.log((N + 1) / (df[t] + 1)) + 1 for t in df}

    tfidf_vecs = []
    for c in chunks:
        tf: Dict[str, float] = {}
        for t in c.tokens:
            tf[t] = tf.get(t, 0) + 1
        norm = math.sqrt(sum((tf[t] * idf.get(t, 1)) ** 2 for t in tf)) or 1
        tfidf_vecs.append({t: (tf[t] * idf.get(t, 1)) / norm for t in tf})

    return chunks, bm25, idf, tfidf_vecs


# ── #2 Hybrid BM25 + TF-IDF retrieval ────────────────────────────────────────
def hybrid_retrieve(chunks, bm25, idf, tfidf_vecs, question: str, top_k: int = 12):
    expanded = expand_query(question)
    q_tokens = tokenize(expanded)
    if not q_tokens:
        return []

    # BM25 scores
    bm25_scores = bm25.get_scores(q_tokens)
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

    # TF-IDF cosine scores
    q_tf: Dict[str, float] = {}
    for t in q_tokens:
        q_tf[t] = q_tf.get(t, 0) + 1
    q_norm = math.sqrt(sum((q_tf[t] * idf.get(t, 1)) ** 2 for t in q_tf)) or 1
    q_vec = {t: (q_tf[t] * idf.get(t, 1)) / q_norm for t in q_tf}

    tfidf_scores = []
    for vec in tfidf_vecs:
        score = sum(q_vec.get(t, 0) * vec.get(t, 0) for t in q_vec)
        tfidf_scores.append(score)
    max_tfidf = max(tfidf_scores) if max(tfidf_scores) > 0 else 1

    # Combine: 60% BM25 + 40% TF-IDF
    combined = [
        0.6 * (bm25_scores[i] / max_bm25) + 0.4 * (tfidf_scores[i] / max_tfidf)
        for i in range(len(chunks))
    ]

    ranked = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)

    # #3 Reranking: boost chunks that contain more query tokens
    seen = set()
    result = []
    for idx in ranked[:top_k * 4]:
        s = combined[idx]
        if s <= 0:
            continue
        c = chunks[idx]
        key = c.text[:100]
        if key in seen:
            continue
        seen.add(key)

        # #3 Rerank bonus: % of query tokens found in chunk
        chunk_token_set = set(c.tokens)
        coverage = sum(1 for t in q_tokens if t in chunk_token_set) / max(len(q_tokens), 1)
        final_score = s * (1 + 0.3 * coverage)

        result.append((final_score, c))
        if len(result) >= top_k * 2:
            break

    # Sort by final reranked score
    result.sort(key=lambda x: x[0], reverse=True)
    return result[:top_k]


# ── #1 LLM con memoria conversazione ─────────────────────────────────────────
def llm_answer(question: str, retrieved, history: List[dict]) -> str:
    api_key = get_api_key()
    if not api_key or OpenAI is None:
        return "⚠️ Chiave API non trovata. Aggiungi OPENAI_API_KEY nei Secrets di Streamlit."

    context = "\n\n".join(f"[Pag. {c.page}] {c.text}" for _, c in retrieved)

    system = (
        "Sei un consulente sindacale UILCOM esperto del Piano sanitario integrativo UniSalute IPZS 2026. "
        "Rispondi agli iscritti in modo chiaro, pratico e preciso.\n\n"
        "REGOLE OBBLIGATORIE:\n"
        "1. Usa SOLO il contenuto fornito come CONTENUTO GUIDA.\n"
        "2. NON riportare mai frasi/estratti della guida verbatim.\n"
        "3. NON inventare massimali, franchigie, scoperti, limiti o procedure non presenti.\n"
        "4. Se il contenuto non permette risposta certa su TUTTI gli aspetti, rispondi comunque "
        "su ciò che trovi e indica solo le parti incerte. "
        "Usa 'Non ho trovato...' solo se il contesto è completamente privo di informazioni rilevanti.\n"
        "5. Se l'utente dice gratis/gratuita, interpreta come: coperta dal piano? con quali condizioni?\n"
        "6. Tieni conto della cronologia della conversazione per capire il contesto.\n"
        "7. Non citare numeri di pagina: li aggiunge l'app.\n\n"
        "FORMATO RISPOSTA:\n"
        "**ESITO:** (Coperta / Parzialmente coperta / Non citata / Serve verifica)\n"
        "**RISPOSTA:** 2–6 righe dirette.\n"
        "**COME FARE:** 3–7 punti operativi (solo se ricavabili dalla guida).\n"
        "**ATTENZIONI:** 0–5 punti su limiti, condizioni, esclusioni (solo se presenti)."
    )

    # Build messages with history (#1 memoria)
    messages = []

    # Add last 4 exchanges for context (not too long)
    for h in history[-8:]:
        if h["role"] in ("user", "assistant"):
            # Strip page references from assistant messages to save tokens
            content = re.sub(r"\n\n---\n📄.*$", "", h["content"], flags=re.DOTALL)
            messages.append({"role": h["role"], "content": content})

    # Current question with context
    messages.append({
        "role": "user",
        "content": f"DOMANDA: {question}\n\nCONTENUTO GUIDA (UNICA FONTE):\n{context}"
    })

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        max_tokens=1000,
        messages=[{"role": "system", "content": system}] + messages,
    )
    return resp.choices[0].message.content.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chatbot UniSalute IPZS 2026", layout="wide", page_icon="🏥")

if not check_password():
    st.stop()

st.title("🏥 Chatbot UniSalute IPZS 2026")
st.caption("Consulente Sindacale UILCOM · Piano sanitario integrativo · Nessun testo guida mostrato")

with st.sidebar:
    st.header("⚙️ Impostazioni")
    top_k    = st.slider("Blocchi recuperati (top-k)", 5, 20, 12, step=1)
    min_hits = st.slider("Soglia minima blocchi", 1, 5, 1)
    strict   = st.toggle("Modalità rigorosa", value=True)
    st.divider()
    if st.button("🗑️ Nuova conversazione"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("v2 · Ricerca ibrida · Memoria conversazione")

# Carica indice
try:
    chunks, bm25, idf, tfidf_vecs = build_index(CHUNKS_PATH)
    st.sidebar.success(f"✅ {len(chunks)} blocchi indicizzati")
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

# Inizializza chat
if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Ciao! Sono il tuo consulente UILCOM per il Piano UniSalute IPZS 2026.\n\nChiedimi pure: *Visita oculistica: è coperta?* oppure *Come faccio a chiedere un rimborso?*"
    }]

# Mostra messaggi
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Suggerimenti rapidi
if len(st.session_state.messages) <= 1:
    st.markdown("**Domande frequenti:**")
    cols = st.columns(2)
    suggestions = [
        "Visita oculistica coperta?",
        "Pulizia denti come fare?",
        "Rimborso fisioterapia?",
        "Impianto dentale previsto?",
        "Ricovero ospedaliero come funziona?",
        "Esami per allergia coperti?",
    ]
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s, key=f"sug_{i}"):
            st.session_state["quick_question"] = s
            st.rerun()

# Gestione domanda rapida
if "quick_question" in st.session_state:
    question = st.session_state.pop("quick_question")
else:
    question = st.chat_input("Scrivi una domanda (es: 'Occhiali: quanto rimborsa?')")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve con ricerca ibrida
    retrieved = hybrid_retrieve(chunks, bm25, idf, tfidf_vecs, question, top_k=top_k)
    all_pages = []
    for _, c in retrieved:
        all_pages.extend(c.pages)
    pages_str = pages_ref_string(all_pages)

    if strict and len(retrieved) < min_hits:
        answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
    else:
        with st.spinner("Analisi in corso..."):
            # Passa la storia per la memoria (#1)
            history = [m for m in st.session_state.messages if m["role"] != "assistant" or len(st.session_state.messages) > 2]
            answer = llm_answer(question, retrieved, history)

    if pages_str:
        answer += f"\n\n---\n📄 **Rif.:** {pages_str}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
