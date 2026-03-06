# app_v3.py — Chatbot UniSalute IPZS 2026 v3
# Aggiunge ricerca semantica con embeddings a tutto ciò che già funziona in v2

import os, re, json, math
from dataclasses import dataclass
from typing import List, Dict
import streamlit as st
from rank_bm25 import BM25Okapi

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

CHUNKS_PATH    = "chunks_data.json"
EMBEDDINGS_PATH = "embeddings.json"

STOPWORDS_IT = {
    "il","lo","la","i","gli","le","un","uno","una","di","a","da","in","su","per","con","tra","fra",
    "e","o","ma","che","come","quanto","quanta","quanti","quante","se","si","no",
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
    "fisioterapia":   ["fisioterapici","riabilitativi","fisioterapia","riabilitazione","trattamenti"],
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
    "sangue":         ["ematologici","diagnostici","accertamenti","esami","laboratorio"],
    "radiografia":    ["radiologica","diagnostica","accertamenti","rx","diagnostici"],
    "ecografia":      ["ecografia","ecocolordoppler","diagnostici","accertamenti"],
    "risonanza":      ["risonanza","magnetica","diagnostici","alta","specializzazione"],
    "tac":            ["tac","diagnostici","alta","specializzazione","accertamenti"],
    "farmaci":        ["medicinali","farmaci","farmacologica","prescritti","curante"],
    "medicine":       ["medicinali","farmaci","prescritti","curante"],
    "rimborso":       ["rimborso","rimborsate","rimborsuale","rimborsi","richiesta"],
    "pagamento":      ["rimborso","liquidate","liquidazione","massimale","franchigia"],
    "convenzionato":  ["convenzionata","convenzionato","strutture","rete"],
    "anziano":        ["autosufficienza","ltc","assistenza"],
    "badante":        ["assistenza","socio","assistenziali","badanti","domiciliare"],
    "psicologia":     ["psichiatrica","psichici","mentali"],
    "termale":        ["termali","terme","termale","cure"],
    "udito":          ["acustiche","udito","protesi","otoemissioni"],
    "ticket":         ["ticket","ssn","nazionale","sanitari","rimborso"],
    "estero":         ["estero","internazionale","rimpatrio","validita","territoriale"],
    "massimale":      ["massimale","limite","annuo","nucleo","familiare"],
    "franchigia":     ["franchigia","scoperto","minimo","indennizzabile"],
    "familiare":      ["familiare","nucleo","coniuge","figli","famiglia"],
    "infortunio":     ["infortunio","infortuni","trauma","pronto","soccorso"],
    "ricovero":       ["ricovero","degenza","ospedaliero","chirurgico","medico"],
}

def expand_query(text):
    lower = text.lower()
    extra = []
    for kw, terms in SYNONYMS.items():
        if kw in lower:
            extra.extend(terms)
    return text + (" " + " ".join(extra) if extra else "")

def tokenize(text):
    parts = re.findall(r"[A-Za-z\xc0-\xd6\xd8-\xf6\xf8-\xff0-9]+", (text or "").lower())
    seen = set(); out = []
    for p in parts:
        if len(p)>=3 and p not in STOPWORDS_IT and p not in seen:
            out.append(p); seen.add(p)
    return out

def cosine(a, b):
    dot = sum(x*y for x,y in zip(a,b))
    na  = math.sqrt(sum(x*x for x in a))
    nb  = math.sqrt(sum(x*x for x in b))
    return dot/(na*nb) if na and nb else 0.0

def pages_ref_string(pages):
    if not pages: return ""
    pages = sorted(set(pages))
    ranges, start, prev = [], pages[0], pages[0]
    for p in pages[1:]:
        if p==prev+1: prev=p
        else: ranges.append((start,prev)); start=prev=p
    ranges.append((start,prev))
    return ", ".join(f"Pag. {a}-{b}" if a!=b else f"Pag. {a}" for a,b in ranges)

def get_api_key():
    key = ""
    if hasattr(st,"secrets"):
        try: key = st.secrets.get("OPENAI_API_KEY","")
        except: pass
    return (key or os.getenv("OPENAI_API_KEY","")).strip()

def check_password():
    correct_pw = ""
    if hasattr(st,"secrets"):
        try: correct_pw = st.secrets.get("APP_PASSWORD","")
        except: pass
    if not correct_pw:
        correct_pw = os.getenv("APP_PASSWORD","Uni79")
    if st.session_state.get("authenticated"):
        return True
    st.markdown("""
    <div style='text-align:center;padding:40px 0 10px 0'>
        <span style='font-size:3rem'>🏥</span>
        <h2>Chatbot UniSalute IPZS 2026 <span style='font-size:1rem;color:#888'>v3</span></h2>
        <p style='color:gray'>Consulente Sindacale UILCOM — Accesso riservato</p>
    </div>""", unsafe_allow_html=True)
    c1,c2,c3 = st.columns([1,2,1])
    with c2:
        pwd = st.text_input("Password", type="password", placeholder="Inserisci la password")
        if st.button("Accedi →", use_container_width=True):
            if pwd == correct_pw:
                st.session_state["authenticated"] = True
                st.rerun()
            else:
                st.error("Password errata. Riprova.")
    return False

@dataclass
class Chunk:
    page: int
    pages: List[int]
    text: str
    tokens: List[str]
    embedding: List[float]

@st.cache_resource
def build_index(chunks_path, embeddings_path):
    with open(chunks_path,"r",encoding="utf-8") as f:
        raw_chunks = json.load(f)
    with open(embeddings_path,"r",encoding="utf-8") as f:
        raw_emb = json.load(f)

    chunks = []
    for i, item in enumerate(raw_emb):
        toks = tokenize(item["text"])
        if toks:
            chunks.append(Chunk(
                page=item["page"],
                pages=item.get("pages",[item["page"]]),
                text=item["text"],
                tokens=toks,
                embedding=item["embedding"]
            ))

    bm25 = BM25Okapi([c.tokens for c in chunks])

    # TF-IDF
    N = len(chunks)
    df = {}
    for c in chunks:
        for t in set(c.tokens): df[t] = df.get(t,0)+1
    idf = {t: math.log((N+1)/(df[t]+1))+1 for t in df}
    tfidf_vecs = []
    for c in chunks:
        tf = {}
        for t in c.tokens: tf[t] = tf.get(t,0)+1
        norm = math.sqrt(sum((tf[t]*idf.get(t,1))**2 for t in tf)) or 1
        tfidf_vecs.append({t:(tf[t]*idf.get(t,1))/norm for t in tf})

    return chunks, bm25, idf, tfidf_vecs

def get_query_embedding(question, api_key):
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=[question])
    return resp.data[0].embedding

def hybrid_semantic_retrieve(chunks, bm25, idf, tfidf_vecs, question, api_key, top_k=12):
    expanded = expand_query(question)
    q_tokens = tokenize(expanded)

    # BM25
    bm25_scores = bm25.get_scores(q_tokens) if q_tokens else [0]*len(chunks)
    max_bm25 = max(bm25_scores) if max(bm25_scores)>0 else 1

    # TF-IDF
    q_tf = {}
    for t in q_tokens: q_tf[t] = q_tf.get(t,0)+1
    q_norm = math.sqrt(sum((q_tf[t]*idf.get(t,1))**2 for t in q_tf)) or 1
    q_vec = {t:(q_tf[t]*idf.get(t,1))/q_norm for t in q_tf}
    tfidf_scores = [sum(q_vec.get(t,0)*vec.get(t,0) for t in q_vec) for vec in tfidf_vecs]
    max_tfidf = max(tfidf_scores) if max(tfidf_scores)>0 else 1

    # Semantic embeddings
    q_emb = get_query_embedding(question, api_key)
    sem_scores = [cosine(q_emb, c.embedding) for c in chunks]
    max_sem = max(sem_scores) if max(sem_scores)>0 else 1

    # Combine: 40% BM25 + 25% TF-IDF + 35% Semantic
    combined = [
        0.40*(bm25_scores[i]/max_bm25) +
        0.25*(tfidf_scores[i]/max_tfidf) +
        0.35*(sem_scores[i]/max_sem)
        for i in range(len(chunks))
    ]

    ranked = sorted(range(len(combined)), key=lambda i: combined[i], reverse=True)
    seen = set(); result = []
    for idx in ranked[:top_k*4]:
        s = combined[idx]
        if s<=0: continue
        c = chunks[idx]
        key = c.text[:100]
        if key in seen: continue
        seen.add(key)
        coverage = sum(1 for t in q_tokens if t in set(c.tokens))/max(len(q_tokens),1)
        result.append((s*(1+0.3*coverage), c))
        if len(result)>=top_k*2: break

    result.sort(key=lambda x: x[0], reverse=True)
    return result[:top_k]

def llm_answer(question, retrieved, history, api_key):
    context = "\n\n".join(f"[Pag. {c.page}] {c.text}" for _,c in retrieved)
    system = (
        "Sei un consulente sindacale UILCOM esperto del Piano sanitario integrativo UniSalute IPZS 2026. "
        "Rispondi agli iscritti in modo chiaro, pratico e preciso.\n\n"
        "REGOLE:\n"
        "1. Usa SOLO il CONTENUTO GUIDA fornito.\n"
        "2. NON copiare frasi dalla guida.\n"
        "3. NON inventare massimali, franchigie, limiti o procedure.\n"
        "4. Rispondi su ciò che trovi; usa 'Non ho trovato...' solo se il contesto è vuoto.\n"
        "5. Gratis/gratuita = coperta dal piano? con quali condizioni?\n"
        "6. Usa la cronologia per capire domande di follow-up.\n"
        "7. Non citare pagine nella risposta.\n\n"
        "FORMATO:\n"
        "**ESITO:** (Coperta / Parzialmente coperta / Non citata / Serve verifica)\n"
        "**RISPOSTA:** 2-6 righe dirette.\n"
        "**COME FARE:** 3-7 punti operativi (solo se nella guida).\n"
        "**ATTENZIONI:** 0-5 punti su limiti/esclusioni (solo se nel testo)."
    )
    messages = []
    for h in history[-8:]:
        if h["role"] in ("user","assistant"):
            content = re.sub(r"\n\n---\n📄.*$","",h["content"],flags=re.DOTALL)
            messages.append({"role":h["role"],"content":content})
    messages.append({"role":"user","content":f"DOMANDA: {question}\n\nCONTENUTO GUIDA:\n{context}"})
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model="gpt-4o-mini", temperature=0.1, max_tokens=1000,
        messages=[{"role":"system","content":system}]+messages)
    return resp.choices[0].message.content.strip()

# ── UI ────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Chatbot UniSalute v3", layout="wide", page_icon="🏥")

if not check_password():
    st.stop()

st.title("🏥 Chatbot UniSalute IPZS 2026")
st.caption("Consulente Sindacale UILCOM · v3 con ricerca semantica · Nessun testo guida mostrato")

with st.sidebar:
    st.header("⚙️ Impostazioni")
    top_k  = st.slider("Blocchi recuperati (top-k)", 5, 20, 12, step=1)
    strict = st.toggle("Modalità rigorosa", value=True)
    st.divider()
    if st.button("🗑️ Nuova conversazione"):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    st.caption("v3 · BM25 + TF-IDF + Semantica · Memoria")

api_key = get_api_key()
if not api_key:
    st.error("⚠️ OPENAI_API_KEY non trovata nei Secrets di Streamlit.")
    st.stop()

try:
    chunks, bm25, idf, tfidf_vecs = build_index(CHUNKS_PATH, EMBEDDINGS_PATH)
    st.sidebar.success(f"✅ {len(chunks)} blocchi indicizzati")
except Exception as e:
    st.error(f"Errore caricamento dati: {e}")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role":"assistant",
        "content":"Ciao! Sono il tuo consulente UILCOM per il Piano UniSalute IPZS 2026 (v3).\n\nChiedimi pure in modo naturale: *Mi sono fatto male allo sport, sono coperto?*"
    }]

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

if len(st.session_state.messages) <= 1:
    st.markdown("**Domande frequenti:**")
    cols = st.columns(2)
    suggestions = [
        "Visita oculistica coperta?","Pulizia denti come fare?",
        "Mi sono fatto male allo sport","Mia madre non riesce a camminare",
        "Ricovero ospedaliero come funziona?","Esami per allergia coperti?",
    ]
    for i,s in enumerate(suggestions):
        if cols[i%2].button(s, key=f"sug_{i}"):
            st.session_state["quick_question"] = s
            st.rerun()

if "quick_question" in st.session_state:
    question = st.session_state.pop("quick_question")
else:
    question = st.chat_input("Scrivi una domanda in modo naturale...")

if question:
    st.session_state.messages.append({"role":"user","content":question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.spinner("Analisi in corso..."):
        try:
            retrieved = hybrid_semantic_retrieve(chunks, bm25, idf, tfidf_vecs, question, api_key, top_k=top_k)
            all_pages = []
            for _,c in retrieved: all_pages.extend(c.pages)
            pages_str = pages_ref_string(all_pages)

            if strict and len(retrieved) < 1:
                answer = "Non ho trovato nella guida UniSalute una risposta certa a questa domanda."
            else:
                answer = llm_answer(question, retrieved, st.session_state.messages[:-1], api_key)

            if pages_str:
                answer += f"\n\n---\n📄 **Rif.:** {pages_str}"
        except Exception as e:
            answer = f"⚠️ Errore: {e}"

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role":"assistant","content":answer})
