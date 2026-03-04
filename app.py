# app.py — Chatbot UniSalute IPZS 2026
# Versione Streamlit Cloud con:
# - Password di accesso
# - Dizionario sinonimi (linguaggio naturale → termini guida)
# - BM25 migliorato (top-k=12)
# - Prompt consulente UILCOM

import os
import re
import json
import hashlib
from dataclasses import dataclass
from typing import List, Tuple

import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from anthropic import Anthropic
except Exception:
    Anthropic = None

# ── Config ────────────────────────────────────────────────────────────────────
CHUNKS_PATH = "chunks_data.json"

STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","si","sì","no",
    "del","dello","della","dei","degli","delle",
    "al","allo","alla","ai","agli","alle",
    "nel","nello","nella","nei","negli","nelle",
    "puo","può","posso","fare","faccio","gratis","gratuita","gratuito","costo","costare",
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
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def normalize_spaces(s: str) -> str:
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
            key = st.secrets.get("ANTHROPIC_API_KEY", "") or st.secrets.get("OPENAI_API_KEY", "")
        except Exception:
            pass
    if not key:
        key = os.getenv("ANTHROPIC_API_KEY", "") or os.getenv("OPENAI_API_KEY", "")
    return (key or "").strip()

def check_password() -> bool:
    """Schermata di login con password."""
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


# ── BM25 Index ────────────────────────────────────────────────────────────────
@st.cache_resource
def build_index(chunks_path: str):
    with open(chunks_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    @dataclass
    class Chunk:
        page: int
        text: str
        tokens: List[str]

    chunks = []
    for item in raw:
        toks = tokenize(item["text"])
        if toks:
            chunks.append(Chunk(page=item["page"], text=item["text"], tokens=toks))

    bm25 = BM25Okapi([c.tokens for c in chunks])
    return chunks, bm25

def retrieve(chunks, bm25, question: str, top_k: int = 12):
    expanded = expand_query(question)
    q_tokens = tokenize(expanded)
    if not q_tokens:
        return []

    scores = bm25.get_scores(q_tokens)
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

    seen = set()
    result = []
    for idx in ranked[:top_k * 3]:
        s = float(scores[idx])
        if s <= 0:
            continue
        c = chunks[idx]
        key = (c.page, c.text[:80])
        if key in seen:
            continue
        seen.add(key)
        result.append((s, c))
        if len(result) >= top_k:
            break
    return result


# ── LLM Answer ────────────────────────────────────────────────────────────────
def llm_answer(question: str, retrieved) -> str:
    api_key = get_api_key()
    if not api_key or Anthropic is None:
        return "⚠️ Chiave API non trovata. Aggiungi ANTHROPIC_API_KEY nei Secrets di Streamlit."

    context = "\n\n".join(f"[Pag. {c.page}] {c.text}" for _, c in retrieved)

    system = (
        "Sei un consulente sindacale UILCOM esperto del Piano sanitario integrativo UniSalute IPZS 2026. "
        "Rispondi agli iscritti in modo chiaro, pratico e preciso.\n\n"
        "REGOLE OBBLIGATORIE:\n"
        "1. Usa SOLO il contenuto fornito come CONTENUTO GUIDA.\n"
        "2. NON riportare mai frasi/estratti della guida verbatim.\n"
        "3. NON inventare massimali, franchigie, scoperti, limiti, condizioni o procedure non presenti.\n"
        "4. Se il contenuto non permette risposta certa su TUTTI gli aspetti, rispondi comunque "
        "su ciò che trovi e indica solo le parti incerte. "
        "Usa 'Non ho trovato...' solo se il contesto è completamente privo di informazioni rilevanti.\n"
        "5. Se l'utente dice gratis/gratuita/gratuito, interpreta come: la prestazione è coperta dal piano? "
        "con quali modalità/condizioni?\n"
        "6. Non citare numeri di pagina: li aggiunge l'app.\n\n"
        "FORMATO RISPOSTA (rispetta sempre questa struttura):\n"
        "**ESITO:** (Coperta / Parzialmente coperta / Non citata / Serve verifica)\n"
        "**RISPOSTA:** 2–6 righe dirette.\n"
        "**COME FARE:** 3–7 punti operativi (solo se ricavabili dalla guida).\n"
        "**ATTENZIONI:** 0–5 punti su limiti, condizioni, esclusioni (solo se presenti nel testo)."
    )

    user = f"DOMANDA: {question}\n\nCONTENUTO GUIDA (UNICA FONTE):\n{context}"

    client = Anthropic(api_key=api_key)
    resp = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1000,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return resp.content[0].text.strip()


# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chatbot UniSalute IPZS 2026", layout="wide", page_icon="🏥")

if not check_password():
    st.stop()

st.title("🏥 Chatbot UniSalute IPZS 2026")
st.caption("Consulente Sindacale UILCOM · Risposte dal piano sanitario integrativo · Nessun testo guida mostrato")

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
    st.caption("Richiede ANTHROPIC_API_KEY nei Secrets di Streamlit.")

# Carica indice
try:
    chunks, bm25 = build_index(CHUNKS_PATH)
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

# Gestione domanda rapida da suggerimento
if "quick_question" in st.session_state:
    question = st.session_state.pop("quick_question")
else:
    question = st.chat_input("Scrivi una domanda (es: 'Occhiali: quanto rimborsa?')")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    retrieved = retrieve(chunks, bm25, question, top_k=top_k)
    pages_str = pages_ref_string([c.page for _, c in retrieved])

    if strict and len(retrieved) < min_hits:
        answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
    else:
        with st.spinner("Analisi in corso..."):
            answer = llm_answer(question, retrieved)

    if pages_str:
        answer += f"\n\n---\n📄 **Rif.:** {pages_str}"

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
    
