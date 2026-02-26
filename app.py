import re
import fitz
import streamlit as st

st.set_page_config(page_title="Guida UniSalute IPZS", layout="wide")

st.title("🔎 Ricerca Guida UniSalute IPZS 2026")
st.write("Inserisci una parola chiave (es: ticket, visita, rimborso, odontoiatria)")

PDF_PATH = "Unis2026.pdf"

query = st.text_input("Parola chiave")

def normalize(text):
    return re.sub(r"\s+", " ", text)

def highlight(text, word):
    return re.sub(
        re.escape(word),
        lambda m: f"<mark>{m.group(0)}</mark>",
        text,
        flags=re.IGNORECASE
    )

if st.button("Cerca") and query:

    doc = fitz.open(PDF_PATH)
    results = []

    for i in range(doc.page_count):
        page = doc.load_page(i)
        text = normalize(page.get_text())

        if re.search(query, text, re.IGNORECASE):
            pos = re.search(query, text, re.IGNORECASE).start()
            start = max(0, pos - 150)
            end = min(len(text), pos + 150)

            snippet = text[start:end]
            results.append((i + 1, snippet))

    if not results:
        st.error("❌ Non presente nella guida")
    else:
        for page, snip in results[:8]:
            st.write(f"📄 Pagina {page}")
            st.markdown(highlight(snip, query), unsafe_allow_html=True)
            st.divider()
