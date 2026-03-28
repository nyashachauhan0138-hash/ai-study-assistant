import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import ollama
import uuid
import json
import re
import io
import textwrap
from datetime import datetime

# ─────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="LumiAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
#  DESIGN SYSTEM  (inspired by getstudymate.com)
#  Clean white/off-white bg · violet/purple brand accent
#  Rounded cards · friendly, legible type
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,400&family=Instrument+Serif:ital@0;1&display=swap');

/* ══════════════  BASE  ══════════════ */
html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: #F7F6FF !important;
    color: #1C1B2E !important;
    -webkit-font-smoothing: antialiased;
}
.block-container {
    max-width: 1080px !important;
    padding: 1.5rem 2rem 4rem !important;
    margin: auto;
}
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F0EFFA; }
::-webkit-scrollbar-thumb { background: #C4BFEF; border-radius: 99px; }

/* ══════════════  SIDEBAR  ══════════════ */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1.5px solid #EAE8F8 !important;
}
[data-testid="stSidebar"] > div { padding: 1.4rem 1.1rem 2rem !important; }

.sb-label {
    font-size: 10px; font-weight: 700; letter-spacing: 1.3px;
    text-transform: uppercase; color: #B5B0D8; margin: 20px 0 8px 2px;
}
.file-pill {
    display: flex; align-items: center; gap: 7px;
    background: #F7F6FF; border: 1px solid #E0DCF5; border-radius: 10px;
    padding: 7px 10px; font-size: 12px; color: #6A66A3; margin-bottom: 5px;
}
.file-pill-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; flex: 1; }
.sb-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; margin: 10px 0; }
.sb-stat {
    background: #F7F6FF; border: 1px solid #E0DCF5; border-radius: 10px;
    padding: 9px 10px; text-align: center;
}
.sb-stat-val {
    font-family: 'Instrument Serif', serif; font-size: 22px; color: #4B3FBF; line-height: 1;
}
.sb-stat-lbl {
    font-size: 10px; color: #B5B0D8; margin-top: 2px;
    text-transform: uppercase; letter-spacing: 0.6px;
}
.sb-footer { margin-top: 22px; padding-top: 14px; border-top: 1px solid #EAE8F8; }
.sb-footer-row {
    display: flex; align-items: center; gap: 6px;
    font-size: 11px; color: #B5B0D8; margin-bottom: 5px;
}
.sb-dot { width: 5px; height: 5px; border-radius: 50%; background: #7C6FE0; flex-shrink: 0; opacity: .5; }

/* ══════════════  BUTTONS  ══════════════ */
.stButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    border-radius: 10px !important; padding: 9px 18px !important;
    border: 1.5px solid #E0DCF5 !important;
    background: #FFFFFF !important; color: #4B3FBF !important;
    transition: all 0.15s !important;
    box-shadow: 0 1px 3px rgba(76,63,191,.06) !important;
}
.stButton > button:hover {
    background: #F0EDFF !important; border-color: #7C6FE0 !important;
    color: #3730A3 !important; transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(76,63,191,.12) !important;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #7C6FE0 0%, #5046C8 100%) !important;
    border: none !important; color: #FFFFFF !important;
    box-shadow: 0 4px 14px rgba(76,63,191,.28) !important;
}
.stButton > button[kind="primary"]:hover {
    background: linear-gradient(135deg, #8B7EEF 0%, #5E52D6 100%) !important;
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(76,63,191,.36) !important;
}

/* ══════════════  INPUTS  ══════════════ */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: #FFFFFF !important; border: 1.5px solid #E0DCF5 !important;
    border-radius: 10px !important; color: #1C1B2E !important;
    font-size: 13px !important; padding: 10px 13px !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #7C6FE0 !important;
    box-shadow: 0 0 0 3px rgba(124,111,224,.14) !important;
}
.stTextInput > label, .stTextArea > label, .stSelectbox > label {
    font-size: 10px !important; font-weight: 700 !important;
    text-transform: uppercase !important; letter-spacing: 1px !important; color: #9490BA !important;
}
.stSelectbox > div > div {
    background: #FFFFFF !important; border: 1.5px solid #E0DCF5 !important;
    border-radius: 10px !important; color: #1C1B2E !important; font-size: 13px !important;
}
.stSelectbox > div > div:focus-within {
    border-color: #7C6FE0 !important;
    box-shadow: 0 0 0 3px rgba(124,111,224,.14) !important;
}
[data-testid="stFileUploader"] {
    background: #F7F6FF !important; border: 1.5px dashed #C4BFEF !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploader"]:hover { border-color: #7C6FE0 !important; }
[data-testid="stFileUploader"] label { color: #9490BA !important; font-size: 13px !important; }

/* ══════════════  TABS  ══════════════ */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #EAE8F8 !important; gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important; font-weight: 600 !important;
    color: #9490BA !important; background: transparent !important;
    border: none !important; padding: 10px 20px !important;
    border-radius: 10px 10px 0 0 !important; transition: color 0.15s;
}
.stTabs [data-baseweb="tab"]:hover { color: #4B3FBF !important; background: #F0EDFF !important; }
.stTabs [aria-selected="true"] {
    color: #4B3FBF !important; border-bottom: 3px solid #7C6FE0 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }
.stTabs [data-baseweb="tab-panel"] { padding: 28px 0 0 !important; }

/* ══════════════  CHAT  ══════════════ */
/* FIX: force dark text in all chat message containers */
[data-testid="stChatMessage"] {
    background: #FFFFFF !important; border: 1.5px solid #EAE8F8 !important;
    border-radius: 16px !important; padding: 14px 18px !important;
    margin-bottom: 10px !important;
    box-shadow: 0 1px 4px rgba(76,63,191,.05) !important;
    color: #1C1B2E !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessage"] div,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] ol,
[data-testid="stChatMessage"] ul,
[data-testid="stChatMessage"] strong,
[data-testid="stChatMessage"] em,
[data-testid="stChatMessage"] code,
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] * {
    color: #1C1B2E !important;
}
[data-testid="stChatInput"] {
    background: #FFFFFF !important; border: 1.5px solid #E0DCF5 !important;
    border-radius: 14px !important;
    box-shadow: 0 2px 8px rgba(76,63,191,.07) !important;
}
[data-testid="stChatInput"]:focus-within {
    border-color: #7C6FE0 !important;
    box-shadow: 0 0 0 3px rgba(124,111,224,.14) !important;
}
[data-testid="stChatInput"] textarea {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 14px !important; color: #1C1B2E !important; background: transparent !important;
}

/* ══════════════  ALERTS  ══════════════ */
.stSuccess { background: #ECFDF5 !important; border: 1px solid #A7F3D0 !important; border-radius: 12px !important; color: #065F46 !important; }
.stInfo    { background: #EEF2FF !important; border: 1px solid #C7D2FE !important; border-radius: 12px !important; color: #3730A3 !important; }
.stWarning { background: #FFFBEB !important; border: 1px solid #FDE68A !important; border-radius: 12px !important; color: #92400E !important; }
.stError   { background: #FFF1F2 !important; border: 1px solid #FECDD3 !important; border-radius: 12px !important; color: #9F1239 !important; }
.stSpinner > div { border-top-color: #7C6FE0 !important; }

/* Download button */
.stDownloadButton > button {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background: linear-gradient(135deg, #7C6FE0 0%, #5046C8 100%) !important;
    border: none !important; border-radius: 10px !important; color: #fff !important;
    font-weight: 600 !important; font-size: 13px !important; padding: 9px 20px !important;
    box-shadow: 0 4px 14px rgba(76,63,191,.25) !important;
}
.stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(76,63,191,.32) !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'Plus Jakarta Sans', sans-serif !important; font-size: 13px !important;
    color: #6A66A3 !important; background: #F7F6FF !important;
    border: 1px solid #E0DCF5 !important; border-radius: 10px !important; padding: 10px 14px !important;
}
.streamlit-expanderContent {
    background: #FDFCFF !important; border: 1px solid #E0DCF5 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important; padding: 14px !important;
}
.stCode, pre {
    background: #F3F1FF !important; border: 1px solid #E0DCF5 !important;
    border-radius: 10px !important; font-size: 12px !important;
}
hr { border-color: #EAE8F8 !important; margin: 16px 0 !important; }
.stRadio [data-testid="stMarkdownContainer"] p { font-size: 13px !important; color: #3D3A66 !important; }
.stRadio > div { gap: 6px !important; }

/* ══════════════  PAGE HEADER  ══════════════ */
/* FIX: prevent brand name from being clipped by sidebar */
.sm-topbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 0 22px 0; border-bottom: 2px solid #EAE8F8; margin-bottom: 28px;
    flex-wrap: nowrap; min-width: 0;
}
.sm-brand { display: flex; align-items: center; gap: 12px; min-width: 0; flex-shrink: 0; }
.sm-brand-icon {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #8B7EEF 0%, #5046C8 100%);
    border-radius: 12px; display: flex; align-items: center; justify-content: center;
    font-size: 20px; box-shadow: 0 4px 12px rgba(76,63,191,.22); flex-shrink: 0;
}
.sm-brand-name {
    font-family: 'Instrument Serif', serif; font-size: 1.6rem;
    color: #1C1B2E; font-style: italic; letter-spacing: -0.3px; white-space: nowrap;
}
.sm-brand-tag { font-size: 11px; color: #9490BA; margin-top: -2px; font-weight: 500; white-space: nowrap; }
.sm-badge {
    display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px;
    border-radius: 99px; font-size: 12px; font-weight: 600;
    background: #F7F6FF; border: 1.5px solid #E0DCF5; color: #9490BA; flex-shrink: 0;
}
.sm-badge.live { background: #ECFDF5; border-color: #6EE7B7; color: #047857; }
.sm-badge-dot { width: 7px; height: 7px; border-radius: 50%; background: #D1D5DB; }
.sm-badge.live .sm-badge-dot { background: #10B981; box-shadow: 0 0 6px #10B98166; }

/* ══════════════  CUSTOM COMPONENTS  ══════════════ */
.sm-empty {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    padding: 52px 24px; border: 2px dashed #DDD9F5; border-radius: 20px;
    text-align: center; background: #FDFCFF; margin: 8px 0;
}
.sm-empty-icon { font-size: 36px; margin-bottom: 14px; }
.sm-empty h4 {
    font-family: 'Instrument Serif', serif; font-size: 1.2rem; font-style: italic;
    color: #4B3FBF; margin: 0 0 8px;
}
.sm-empty p { font-size: 13px; color: #9490BA; max-width: 300px; line-height: 1.65; margin: 0; }

.source-card {
    padding: 11px 14px; border-radius: 12px; background: #F7F6FF;
    border: 1px solid #E0DCF5; margin-bottom: 8px;
    font-size: 12px; color: #6A66A3; line-height: 1.6; transition: border-color 0.15s;
}
.source-card:hover { border-color: #AEA5E8; }
.source-card b { color: #5046C8; display: block; margin-bottom: 4px; font-size: 11px; }

.feature-banner {
    background: linear-gradient(135deg, #EEE9FF 0%, #E8E3FF 100%);
    border: 1px solid #D4CCFF; border-radius: 16px;
    padding: 16px 20px; display: flex; align-items: center; gap: 14px; margin-bottom: 22px;
}
.feature-banner-icon { font-size: 28px; flex-shrink: 0; }
.feature-banner h4 {
    font-family: 'Instrument Serif', serif; font-size: 1rem; font-style: italic;
    color: #4B3FBF; margin: 0 0 2px;
}
.feature-banner p { font-size: 12px; color: #7C6FE0; margin: 0; }

.stat-row { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; margin-bottom: 24px; }
.stat-card {
    background: #FFFFFF; border: 1.5px solid #EAE8F8; border-radius: 16px;
    padding: 18px 16px; box-shadow: 0 1px 4px rgba(76,63,191,.05);
}
.stat-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: #B5B0D8; margin-bottom: 6px; }
.stat-value { font-family: 'Instrument Serif', serif; font-size: 2.2rem; color: #4B3FBF; line-height: 1; }
.stat-value.green { color: #059669; }
.stat-value.text { font-size: 14px; margin-top: 4px; font-family: 'Plus Jakarta Sans', sans-serif; font-weight: 600; }

.doc-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 12px; }
.doc-card {
    background: #FFFFFF; border: 1.5px solid #EAE8F8; border-radius: 16px; padding: 18px 16px;
    transition: border-color 0.15s, box-shadow 0.15s, transform 0.15s;
    box-shadow: 0 1px 4px rgba(76,63,191,.05);
}
.doc-card:hover { border-color: #AEA5E8; box-shadow: 0 4px 16px rgba(76,63,191,.12); transform: translateY(-2px); }
.doc-card-icon { font-size: 26px; margin-bottom: 10px; }
.doc-card h4 { font-size: 13px; font-weight: 600; color: #1C1B2E; margin: 0 0 4px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.doc-card p { font-size: 11px; color: #B5B0D8; margin: 0; }

.quiz-card {
    background: #FFFFFF; border: 1.5px solid #EAE8F8; border-radius: 18px;
    padding: 22px 22px 16px; margin-bottom: 16px;
    box-shadow: 0 1px 4px rgba(76,63,191,.05);
}
.quiz-q-num { font-size: 10px; font-weight: 700; letter-spacing: 1.2px; text-transform: uppercase; color: #AEA5E8; margin-bottom: 8px; }
.quiz-q-num span { background: #EEE9FF; color: #7C6FE0; padding: 2px 9px; border-radius: 99px; font-size: 10px; }
.quiz-q-text { font-size: 15px; font-weight: 500; color: #1C1B2E; line-height: 1.65; margin-bottom: 18px; }
.opt-correct { padding: 8px 14px; border-radius: 10px; font-size: 13px; background: #ECFDF5; border: 1.5px solid #6EE7B7; color: #065F46; margin: 5px 0; font-weight: 500; }
.opt-wrong   { padding: 8px 14px; border-radius: 10px; font-size: 13px; background: #FFF1F2; border: 1.5px solid #FECDD3; color: #9F1239; margin: 5px 0; font-weight: 500; }
.opt-neutral { padding: 8px 14px; border-radius: 10px; font-size: 13px; background: #F7F6FF; border: 1.5px solid #E0DCF5; color: #9490BA; margin: 5px 0; }
.quiz-explanation { background: #EEF2FF; border: 1px solid #C7D2FE; border-left: 3px solid #7C6FE0; border-radius: 0 10px 10px 0; padding: 10px 14px; margin-top: 12px; font-size: 12px; color: #3730A3; line-height: 1.65; }

.score-banner { background: linear-gradient(135deg,#F0EDFF,#EEF2FF); border: 2px solid #C4BFEF; border-radius: 20px; padding: 28px 24px; text-align: center; margin-bottom: 24px; }
.score-banner .score-num { font-family: 'Instrument Serif', serif; font-size: 3.2rem; color: #4B3FBF; line-height: 1; }
.score-banner .score-sub { font-size: 13px; color: #9490BA; margin-top: 6px; }
.score-banner.score-great { background: linear-gradient(135deg,#ECFDF5,#D1FAE5); border-color: #6EE7B7; }
.score-banner.score-great .score-num { color: #065F46; }
.score-banner.score-ok   { background: linear-gradient(135deg,#FFFBEB,#FEF3C7); border-color: #FDE68A; }
.score-banner.score-ok   .score-num { color: #92400E; }
.score-banner.score-low  { background: linear-gradient(135deg,#FFF1F2,#FFE4E6); border-color: #FECDD3; }
.score-banner.score-low  .score-num { color: #9F1239; }

.progress-bar-wrap { background: #EAE8F8; border-radius: 99px; height: 6px; margin-bottom: 14px; }
.progress-bar-fill { background: linear-gradient(90deg,#7C6FE0,#5046C8); border-radius: 99px; height: 6px; transition: width 0.3s; }
.progress-label { font-size: 12px; color: #9490BA; margin-bottom: 6px; }

.notes-info-bar { display: flex; align-items: center; justify-content: space-between; padding: 12px 18px; background: #F7F6FF; border: 1px solid #E0DCF5; border-radius: 12px; margin-bottom: 14px; font-size: 12px; color: #9490BA; gap: 10px; flex-wrap: wrap; }
.notes-info-bar .tag { background: #EEE9FF; color: #7C6FE0; padding: 2px 9px; border-radius: 99px; font-size: 11px; font-weight: 600; }

.notes-preview { background: #FFFFFF; color: #1C1B2E; border-radius: 20px; padding: 40px 48px; font-family: 'Plus Jakarta Sans', sans-serif; line-height: 1.8; border: 1.5px solid #EAE8F8; box-shadow: 0 4px 24px rgba(76,63,191,.07); margin-top: 8px; }
.notes-preview h1 { font-family: 'Instrument Serif', serif; font-size: 1.75rem; font-style: italic; color: #1C1B2E; border-bottom: 2px solid #7C6FE0; padding-bottom: 12px; margin-bottom: 6px; }
.notes-preview h2 { font-size: 0.78rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1.2px; color: #7C6FE0; margin-top: 28px; margin-bottom: 8px; }
.notes-preview p { color: #3D3A66; margin-bottom: 10px; font-size: 0.93rem; }
.notes-preview ul { padding-left: 18px; color: #3D3A66; }
.notes-preview li { margin-bottom: 5px; font-size: 0.92rem; line-height: 1.7; }
.notes-preview .key-term { background: #EEE9FF; border-left: 3px solid #7C6FE0; padding: 6px 14px; margin: 6px 0; border-radius: 0 8px 8px 0; font-size: 0.91rem; }
.notes-preview .summary-box { background: #ECFDF5; border: 1px solid #A7F3D0; border-radius: 12px; padding: 14px 18px; margin: 18px 0; font-size: 0.91rem; color: #065F46; line-height: 1.7; }
.notes-meta { font-size: 11px; color: #B5B0D8; margin-bottom: 22px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  LOAD RESOURCES
# ─────────────────────────────────────────
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_db():
    client = PersistentClient(path="chroma_db")
    return client, client.get_or_create_collection("studymate_docs")

model = load_model()
chroma_client, collection = load_db()


# ─────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

def extract_text_pymupdf(file):
    file.seek(0)
    raw = file.read()
    doc = fitz.open(stream=raw, filetype="pdf")
    text = "".join(page.get_text() + "\n" for page in doc)
    doc.close()
    return text

def process_pdfs(files):
    all_chunks, all_embeddings, all_ids, all_metadata = [], [], [], []
    for file in files:
        text = extract_text_pymupdf(file)
        if not text.strip():
            st.warning(f"Could not extract text from {file.name}.")
            continue
        chunks = chunk_text(text)
        embeddings = model.encode(chunks).tolist()
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": file.name}] * len(chunks)
        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        all_metadata.extend(metadata)
    if all_chunks:
        collection.add(documents=all_chunks, embeddings=all_embeddings,
                       ids=all_ids, metadatas=all_metadata)
        return len(all_chunks)
    return 0

def clear_documents():
    chroma_client.delete_collection("studymate_docs")
    chroma_client.get_or_create_collection("studymate_docs")
    st.cache_resource.clear()

def retrieve_context(question, n_results=6):
    embedding = model.encode([question])[0].tolist()
    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(n_results, max(collection.count(), 1))
    )
    return [(d, m) for d, m in zip(results["documents"][0], results["metadatas"][0]) if d.strip()]

def build_prompt(context_text, question):
    return f"""You are LumiAI, a friendly and expert AI academic assistant.

STRICT RULES:
1. Answer ONLY using the provided context from the student's documents.
2. If clearly present → give a clear, well-structured explanation.
3. If NOT in the context → say: "I couldn't find this in your uploaded documents."
4. Do NOT use outside knowledge or make things up.
5. Be concise but thorough. Use bullet points for multi-step concepts.

Context:
{context_text}

Student's Question: {question}
Answer:"""


# ─────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────
for key, default in {
    "messages": [], "docs_processed": False,
    "quiz_questions": [], "quiz_answers": {}, "quiz_submitted": False, "quiz_topic": "",
    "notes_content": None, "notes_topic": "", "notes_pdf_bytes": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding-bottom:16px;border-bottom:1.5px solid #EAE8F8;margin-bottom:4px;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#8B7EEF,#5046C8);border-radius:10px;
             display:flex;align-items:center;justify-content:center;font-size:18px;
             box-shadow:0 3px 10px rgba(76,63,191,.25);flex-shrink:0;">🎓</div>
        <div>
            <div style="font-family:'Instrument Serif',serif;font-size:1.15rem;font-style:italic;color:#1C1B2E;line-height:1.1;">LumiAI</div>
            <div style="font-size:10px;color:#B5B0D8;letter-spacing:.3px;">AI Study Assistant</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sb-label">Study Materials</div>', unsafe_allow_html=True)

    uploaded_files = st.file_uploader(
        "Upload PDFs", type="pdf", accept_multiple_files=True,
        label_visibility="collapsed", help="Upload textbooks, lecture notes, or research papers",
    )

    if uploaded_files:
        for f in uploaded_files:
            kb = round(f.size / 1024, 1)
            st.markdown(f"""
            <div class="file-pill">
                <span style="font-size:14px;">📄</span>
                <span class="file-pill-name">{f.name}</span>
                <span style="font-size:10px;color:#C4BFEF;flex-shrink:0;">{kb} KB</span>
            </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("⚡ Process", use_container_width=True, type="primary"):
            if uploaded_files:
                with st.spinner("Indexing…"):
                    count = process_pdfs(uploaded_files)
                if count:
                    st.success(f"✓ {count} chunks ready")
                    st.session_state.docs_processed = True
            else:
                st.warning("Upload PDFs first.")
    with c2:
        if st.button("🗑️ Clear", use_container_width=True):
            clear_documents()
            st.session_state.docs_processed = False
            st.success("Cleared!")

    st.markdown('<div class="sb-label">Session</div>', unsafe_allow_html=True)
    if st.button("💬 New Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    try:
        chunk_count = collection.count()
        doc_count = len({m["source"] for m in collection.get(limit=500, include=["metadatas"])["metadatas"]}) if chunk_count > 0 else 0
    except:
        chunk_count = doc_count = 0

    st.markdown(f"""
    <div class="sb-stats">
        <div class="sb-stat"><div class="sb-stat-val">{doc_count}</div><div class="sb-stat-lbl">Docs</div></div>
        <div class="sb-stat"><div class="sb-stat-val">{chunk_count}</div><div class="sb-stat-lbl">Chunks</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="sb-footer">
        <div class="sb-footer-row"><div class="sb-dot"></div>Llama 3.1 via Ollama</div>
        <div class="sb-footer-row"><div class="sb-dot"></div>MiniLM-L6-v2 embeddings</div>
        <div class="sb-footer-row"><div class="sb-dot"></div>ChromaDB vector store</div>
        <div class="sb-footer-row"><div class="sb-dot"></div>PyMuPDF PDF engine</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────
#  PAGE HEADER
# ─────────────────────────────────────────
is_live = collection.count() > 0
st.markdown(f"""
<div class="sm-topbar">
    <div class="sm-brand">
        <div class="sm-brand-icon">🎓</div>
        <div>
            <div class="sm-brand-name">LumiAI</div>
            <div class="sm-brand-tag">Your AI-powered study companion</div>
        </div>
    </div>
    <div class="sm-badge {'live' if is_live else ''}">
        <div class="sm-badge-dot"></div>
        {'Index live' if is_live else 'No documents'}
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["💬  Chat", "📂  Documents", "🧠  Quiz", "📝  Notes"])


# ═══════════════════════════════════════════════════
#  TAB 1 — CHAT
# ═══════════════════════════════════════════════════
with tab1:
    st.markdown("""
    <div class="feature-banner">
        <div class="feature-banner-icon">💬</div>
        <div>
            <h4>Ask anything from your documents</h4>
            <p>Answers are always grounded in your uploaded PDFs — no hallucinations.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown("""
        <div class="sm-empty">
            <div class="sm-empty-icon">📖</div>
            <h4>Ready to start studying?</h4>
            <p>Upload your PDFs in the sidebar, click Process, then ask anything about your material.</p>
        </div>""", unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask a question about your documents…")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching your documents…"):
                pairs = retrieve_context(question)

            if not pairs:
                answer = "⚠️ No documents processed yet. Upload and process your PDFs first."
                st.markdown(answer)
            else:
                context_text = "\n\n".join([doc for doc, _ in pairs])
                prompt = build_prompt(context_text, question)
                response = ollama.chat(model="llama3.1",
                                       messages=[{"role": "user", "content": prompt}],
                                       stream=True)
                full = ""
                placeholder = st.empty()
                for chunk in response:
                    full += chunk["message"]["content"]
                    placeholder.markdown(full)
                answer = full

                st.markdown("---")
                st.markdown('<p style="font-size:10px;font-weight:700;letter-spacing:1.1px;text-transform:uppercase;color:#B5B0D8;margin-bottom:8px;">Sources used</p>', unsafe_allow_html=True)
                seen = {}
                for doc, meta in pairs:
                    seen.setdefault(meta["source"], []).append(doc[:220].replace("\n", " ") + "…")
                for src, previews in seen.items():
                    st.markdown(f'<div class="source-card"><b>📄 {src}</b>{" … ".join(previews[:2])}</div>', unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer})


# ═══════════════════════════════════════════════════
#  TAB 2 — DOCUMENTS
# ═══════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div class="feature-banner">
        <div class="feature-banner-icon">📂</div>
        <div>
            <h4>Your indexed study materials</h4>
            <p>All processed PDFs are stored in the vector database and ready to query.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    try:
        count = collection.count()
        if count > 0:
            sample = collection.get(limit=500, include=["metadatas"])
            sources = list({m["source"] for m in sample["metadatas"]})

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card"><div class="stat-label">Indexed chunks</div><div class="stat-value">{count}</div></div>
                <div class="stat-card"><div class="stat-label">Documents</div><div class="stat-value">{len(sources)}</div></div>
                <div class="stat-card"><div class="stat-label">Status</div><div class="stat-value green text">Ready ✓</div></div>
            </div>""", unsafe_allow_html=True)

            st.markdown('<div class="doc-grid">', unsafe_allow_html=True)
            for s in sources:
                st.markdown(f'<div class="doc-card"><div class="doc-card-icon">📄</div><h4>{s}</h4><p>Indexed · searchable</p></div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("")
            st.info("To remove all documents, use the **🗑️ Clear** button in the sidebar.")
        else:
            st.markdown("""
            <div class="sm-empty">
                <div class="sm-empty-icon">📂</div>
                <h4>No documents yet</h4>
                <p>Upload PDFs from the sidebar and click ⚡ Process to index them.</p>
            </div>""", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not read database: {e}")


# ─────────────────────────────────────────
#  QUIZ HELPERS
# ─────────────────────────────────────────
def generate_quiz(topic, num_questions, difficulty):
    query = topic if topic.strip() else "key concepts and important ideas"
    pairs = retrieve_context(query, n_results=8)
    if not pairs:
        return None, "No documents indexed. Please upload and process PDFs first."
    context_text = "\n\n".join([doc for doc, _ in pairs])
    difficulty_desc = {"Easy": "straightforward recall", "Medium": "application of concepts", "Hard": "analysis and synthesis"}[difficulty]

    prompt = f"""Generate exactly {num_questions} MCQ questions from the context.
Difficulty: {difficulty} — {difficulty_desc}.
Output ONLY valid JSON array — no markdown, no preamble:
[{{"question":"...","options":{{"A":"...","B":"...","C":"...","D":"..."}},"answer":"A","explanation":"..."}}]
Context:\n{context_text}\nTopic: {topic or 'general concepts'}\nJSON:"""

    response = ollama.chat(model="llama3.1", messages=[{"role":"user","content":prompt}], stream=False)
    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw), None
    except:
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group()), None
            except: pass
        return None, "Could not parse quiz. Try again."


# ═══════════════════════════════════════════════════
#  TAB 3 — QUIZ
# ═══════════════════════════════════════════════════
with tab3:
    st.markdown("""
    <div class="feature-banner">
        <div class="feature-banner-icon">🧠</div>
        <div>
            <h4>Test your knowledge</h4>
            <p>AI-generated multiple choice questions, pulled directly from your documents.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        quiz_topic = st.text_input("Topic", placeholder="e.g. photosynthesis, Newton's laws, WWII…")
    with col_b:
        num_q = st.selectbox("Questions", [3, 5, 8, 10], index=1)
    with col_c:
        difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)

    col_gen, col_reset = st.columns([2, 1])
    with col_gen:
        gen_clicked = st.button("⚡ Generate Quiz", type="primary", use_container_width=True)
    with col_reset:
        if st.button("↺ Reset", use_container_width=True):
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_topic = ""
            st.rerun()

    if gen_clicked:
        with st.spinner("Generating quiz from your documents…"):
            questions, error = generate_quiz(quiz_topic, num_q, difficulty)
        if error:
            st.error(f"❌ {error}")
        elif questions:
            st.session_state.quiz_questions = questions
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_topic = quiz_topic
            st.rerun()
        else:
            st.error("❌ No questions generated. Try again.")

    if not st.session_state.quiz_questions:
        st.markdown("""
        <div class="sm-empty">
            <div class="sm-empty-icon">🧠</div>
            <h4>No quiz yet</h4>
            <p>Set a topic and difficulty, then click Generate Quiz to get started.</p>
        </div>""", unsafe_allow_html=True)
    else:
        questions = st.session_state.quiz_questions

        if st.session_state.quiz_submitted:
            correct = sum(1 for i, q in enumerate(questions) if st.session_state.quiz_answers.get(i) == q.get("answer"))
            total = len(questions)
            pct = int(correct / total * 100)
            cls = "score-great" if pct >= 80 else ("score-ok" if pct >= 60 else "score-low")
            emoji = "🏆" if pct >= 80 else ("👍" if pct >= 60 else "📖")
            msg = "Excellent!" if pct >= 80 else ("Good work!" if pct >= 60 else "Keep studying!")
            st.markdown(f"""
            <div class="score-banner {cls}">
                <div style="font-size:2.4rem;margin-bottom:6px;">{emoji}</div>
                <div class="score-num">{correct}/{total}</div>
                <div class="score-sub">{pct}% correct · {msg}</div>
            </div>""", unsafe_allow_html=True)
        else:
            answered = len(st.session_state.quiz_answers)
            pct_done = int(answered / len(questions) * 100)
            st.markdown(f'<div class="progress-label">{answered} of {len(questions)} answered</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="progress-bar-wrap"><div class="progress-bar-fill" style="width:{pct_done}%"></div></div>', unsafe_allow_html=True)

        st.markdown("---")

        for i, q in enumerate(questions):
            options_dict = q.get("options", {})
            option_labels = [f"{k}. {v}" for k, v in options_dict.items()]
            option_keys = list(options_dict.keys())

            st.markdown(f"""
            <div class="quiz-card">
                <div class="quiz-q-num">Question <span>{i+1} of {len(questions)}</span></div>
                <div class="quiz-q-text">{q.get("question","")}</div>
            </div>""", unsafe_allow_html=True)

            if not st.session_state.quiz_submitted:
                selected = st.radio(f"q{i}", option_labels, index=None, key=f"radio_{i}", label_visibility="collapsed")
                if selected:
                    st.session_state.quiz_answers[i] = option_keys[option_labels.index(selected)]
            else:
                correct_key = q.get("answer", "")
                user_key = st.session_state.quiz_answers.get(i)
                for k, v in options_dict.items():
                    label = f"{k}. {v}"
                    if k == correct_key:
                        st.markdown(f'<div class="opt-correct">✓ {label}</div>', unsafe_allow_html=True)
                    elif k == user_key and user_key != correct_key:
                        st.markdown(f'<div class="opt-wrong">✗ {label} <em style="opacity:.7">(your answer)</em></div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="opt-neutral">{label}</div>', unsafe_allow_html=True)
                if "explanation" in q:
                    st.markdown(f'<div class="quiz-explanation">💡 <strong>Explanation:</strong> {q["explanation"]}</div>', unsafe_allow_html=True)
            st.markdown("")

        if not st.session_state.quiz_submitted:
            if st.button("Submit Answers →", type="primary", use_container_width=True):
                st.session_state.quiz_submitted = True
                st.rerun()


# ─────────────────────────────────────────
#  NOTES HELPERS
# ─────────────────────────────────────────
def generate_notes_content(topic, style, detail_level):
    query = topic if topic.strip() else "main concepts and key ideas"
    n_chunks = {"Brief": 5, "Standard": 10, "Detailed": 16}[detail_level]
    pairs = retrieve_context(query, n_results=n_chunks)
    if not pairs:
        return None, "No documents indexed. Please upload and process PDFs first."

    context_text = "\n\n".join([doc for doc, _ in pairs])
    sources = list({m["source"] for _, m in pairs})
    style_desc = {"Structured": "sections with headings and bullets", "Cornell Style": "main notes + cues + summary", "Mind Map Outline": "hierarchical outline", "Exam Cram": "condensed key facts"}[style]
    detail_desc = {"Brief": "concise, high-level", "Standard": "balanced with key details", "Detailed": "comprehensive"}[detail_level]

    prompt = f"""Generate structured study notes. Style: {style} ({style_desc}). Detail: {detail_level} ({detail_desc}).
Output ONLY valid JSON (no markdown, no preamble):
{{"title":"...","summary":"2-3 sentence overview","sections":[{{"heading":"...","content":"...","bullets":["..."]}}],"key_terms":[{{"term":"...","definition":"..."}}],"exam_tips":["..."],"sources":{json.dumps(sources)},"style":"{style}","detail":"{detail_level}","generated_at":"{datetime.now().strftime('%b %d, %Y %H:%M')}"}}
Context:\n{context_text}\nJSON:"""

    response = ollama.chat(model="llama3.1", messages=[{"role":"user","content":prompt}], stream=False)
    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()
    try:
        return json.loads(raw), None
    except:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try: return json.loads(m.group()), None
            except: pass
        return None, "Could not parse notes. Try again."


def build_notes_pdf(notes_data):
    from reportlab.lib.pagesizes import letter
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer,
                                     Table, TableStyle, HRFlowable, KeepTogether)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_CENTER

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            rightMargin=.75*inch, leftMargin=.75*inch,
                            topMargin=.75*inch, bottomMargin=.75*inch)
    ACCENT   = colors.HexColor('#7C6FE0')
    MUTED    = colors.HexColor('#9490BA')
    DARK     = colors.HexColor('#1C1B2E')
    GREEN_BG = colors.HexColor('#ECFDF5')
    GREEN_BD = colors.HexColor('#6EE7B7')
    LILA_BG  = colors.HexColor('#EEE9FF')
    AMBER_BG = colors.HexColor('#FFFBEB')

    styles = getSampleStyleSheet()
    ts = lambda name, **kw: ParagraphStyle(name, parent=styles['Normal'], **kw)
    title_s  = ts('ti', fontName='Helvetica-BoldOblique', fontSize=20, textColor=DARK, spaceAfter=4, leading=26)
    meta_s   = ts('me', fontName='Helvetica', fontSize=9, textColor=MUTED, spaceAfter=16)
    h2_s     = ts('h2', fontName='Helvetica-Bold', fontSize=9, textColor=ACCENT, spaceBefore=14, spaceAfter=6, leading=14)
    body_s   = ts('bo', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#3D3A66'), spaceAfter=6, leading=15)
    bullet_s = ts('bl', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#3D3A66'), leftIndent=16, spaceAfter=3, leading=14)
    sum_s    = ts('su', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#065F46'), leading=15)
    term_s   = ts('tr', fontName='Helvetica-Bold', fontSize=10, textColor=colors.HexColor('#4B3FBF'), leading=14)
    def_s    = ts('de', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#3D3A66'), leading=14)
    tip_s    = ts('tp', fontName='Helvetica', fontSize=10, textColor=colors.HexColor('#92400E'), spaceAfter=4, leading=14)
    footer_s = ts('fo', fontName='Helvetica', fontSize=8, textColor=MUTED, alignment=TA_CENTER, spaceBefore=4)

    story = []
    title  = notes_data.get("title", "Study Notes")
    gen_at = notes_data.get("generated_at", "")
    sl     = notes_data.get("style", "")
    dl     = notes_data.get("detail", "")
    srcs   = notes_data.get("sources", [])
    src_str= ", ".join(srcs) if srcs else "your documents"

    story.append(Paragraph(title, title_s))
    story.append(Paragraph(f"Generated {gen_at} · {sl} · {dl} · {src_str}", meta_s))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT, spaceAfter=14))

    summary = notes_data.get("summary","")
    if summary:
        t = Table([[Paragraph(f"<b>Overview:</b> {summary}", sum_s)]], colWidths=[6.5*inch])
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),GREEN_BG),('BOX',(0,0),(-1,-1),1,GREEN_BD),
            ('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10),
            ('LEFTPADDING',(0,0),(-1,-1),14),('RIGHTPADDING',(0,0),(-1,-1),14)]))
        story.extend([t, Spacer(1,14)])

    for sec in notes_data.get("sections",[]):
        blk = [Paragraph(sec.get("heading",""), h2_s)]
        if sec.get("content"): blk.append(Paragraph(sec["content"], body_s))
        for b in sec.get("bullets",[]): blk.append(Paragraph(f"• {b}", bullet_s))
        blk.append(Spacer(1,4))
        story.append(KeepTogether(blk))

    story.extend([Spacer(1,8), HRFlowable(width="100%",thickness=1,color=colors.HexColor('#EAE8F8'),spaceAfter=12)])

    kt = notes_data.get("key_terms",[])
    if kt:
        story.append(Paragraph("Key Terms", h2_s))
        rows = [[Paragraph(k["term"],term_s), Paragraph(k["definition"],def_s)] for k in kt]
        tbl = Table(rows, colWidths=[1.8*inch,4.7*inch])
        tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(0,-1),LILA_BG),('VALIGN',(0,0),(-1,-1),'TOP'),
            ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
            ('LEFTPADDING',(0,0),(-1,-1),10),('RIGHTPADDING',(0,0),(-1,-1),10),
            ('ROWBACKGROUNDS',(1,0),(1,-1),[colors.white,colors.HexColor('#F7F6FF')]),
            ('GRID',(0,0),(-1,-1),.4,colors.HexColor('#E0DCF5'))]))
        story.extend([tbl, Spacer(1,14)])

    tips = notes_data.get("exam_tips",[])
    if tips:
        rows2 = [[Paragraph("<b>Exam Tips</b>", h2_s)]] + [[Paragraph(f"⚡ {t}", tip_s)] for t in tips]
        tbl2 = Table(rows2, colWidths=[6.5*inch])
        tbl2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),AMBER_BG),('BOX',(0,0),(-1,-1),1,colors.HexColor('#FDE68A')),
            ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
            ('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12)]))
        story.extend([tbl2, Spacer(1,16)])

    story.extend([HRFlowable(width="100%",thickness=1,color=colors.HexColor('#EAE8F8'),spaceAfter=6),
                  Paragraph("Generated by LumiAI · For personal study use only", footer_s)])
    doc.build(story)
    buf.seek(0)
    return buf.read()


def render_notes_preview(notes_data):
    title     = notes_data.get("title","Study Notes")
    summary   = notes_data.get("summary","")
    sections  = notes_data.get("sections",[])
    key_terms = notes_data.get("key_terms",[])
    exam_tips = notes_data.get("exam_tips",[])
    sources   = notes_data.get("sources",[])
    gen_at    = notes_data.get("generated_at","")

    secs_html = ""
    for sec in sections:
        bullets = "".join(f"<li>{b}</li>" for b in sec.get("bullets",[]))
        c = sec.get("content","")
        secs_html += f'<h2>{sec.get("heading","")}</h2>{"<p>"+c+"</p>" if c else ""}{"<ul>"+bullets+"</ul>" if bullets else ""}'

    terms_html = "".join(f'<div class="key-term"><b>{kt["term"]}</b>: {kt["definition"]}</div>' for kt in key_terms)
    tips_html  = "".join(f"<li>⚡ {t}</li>" for t in exam_tips)
    src_str    = " · ".join(sources) if sources else "your documents"

    st.markdown(f"""
    <div class="notes-preview">
        <h1>{title}</h1>
        <div class="notes-meta">Generated {gen_at} &nbsp;·&nbsp; Source: {src_str}</div>
        {"<div class='summary-box'>📋 <b>Overview:</b> "+summary+"</div>" if summary else ""}
        {secs_html}
        {"<h2>Key Terms</h2>"+terms_html if terms_html else ""}
        {"<h2>Exam Tips</h2><ul>"+tips_html+"</ul>" if tips_html else ""}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════
#  TAB 4 — NOTES
# ═══════════════════════════════════════════════════
with tab4:
    st.markdown("""
    <div class="feature-banner">
        <div class="feature-banner-icon">📝</div>
        <div>
            <h4>Generate beautiful study notes</h4>
            <p>Structured notes from your documents, exportable as a formatted PDF.</p>
        </div>
    </div>""", unsafe_allow_html=True)

    if collection.count() == 0:
        st.markdown("""
        <div class="sm-empty">
            <div class="sm-empty-icon">📝</div>
            <h4>No documents indexed yet</h4>
            <p>Upload and process your PDFs first, then generate structured study notes.</p>
        </div>""", unsafe_allow_html=True)
    else:
        col_a, col_b, col_c = st.columns([3, 1.5, 1.5])
        with col_a:
            notes_topic = st.text_input("Topic", placeholder="e.g. Calvin Cycle, Newton's Laws, WW2…")
        with col_b:
            notes_style = st.selectbox("Note Style", ["Structured","Cornell Style","Mind Map Outline","Exam Cram"])
        with col_c:
            detail_level = st.selectbox("Detail Level", ["Brief","Standard","Detailed"], index=1)

        col_gen, col_clr = st.columns([2, 1])
        with col_gen:
            gen_notes = st.button("✨ Generate Notes", type="primary", use_container_width=True)
        with col_clr:
            if st.button("↺ Clear", use_container_width=True):
                st.session_state.notes_content = None
                st.session_state.notes_pdf_bytes = None
                st.session_state.notes_topic = ""
                st.rerun()

        if gen_notes:
            with st.spinner("Reading your documents and composing notes…"):
                notes_data, error = generate_notes_content(notes_topic, notes_style, detail_level)
            if error:
                st.error(f"❌ {error}")
            elif notes_data:
                with st.spinner("Building PDF…"):
                    try:
                        pdf_bytes = build_notes_pdf(notes_data)
                        st.session_state.notes_pdf_bytes = pdf_bytes
                    except Exception as e:
                        st.warning(f"PDF generation failed: {e}")
                        st.session_state.notes_pdf_bytes = None
                st.session_state.notes_content = notes_data
                st.session_state.notes_topic = notes_topic
                st.rerun()

        if st.session_state.notes_content:
            nd = st.session_state.notes_content
            topic_label = st.session_state.notes_topic or "All Concepts"

            st.markdown(f"""
            <div class="notes-info-bar">
                <span>
                    <span class="tag">{topic_label}</span>
                    &nbsp; {nd.get('style','')} &nbsp;·&nbsp; {nd.get('detail','')}
                    &nbsp;·&nbsp; {len(nd.get('sections',[]))} sections &nbsp;·&nbsp;
                    {len(nd.get('key_terms',[]))} key terms
                </span>
            </div>""", unsafe_allow_html=True)

            if st.session_state.notes_pdf_bytes:
                fname = (notes_topic or "study_notes").replace(" ","_").lower()
                st.download_button(label="⬇️ Download PDF", data=st.session_state.notes_pdf_bytes,
                                   file_name=f"{fname}_notes.pdf", mime="application/pdf", type="primary")

            render_notes_preview(nd)

            with st.expander("📋 Copy as Markdown"):
                md = [f"# {nd.get('title','Notes')}\n"]
                if nd.get("summary"): md.append(f"> {nd['summary']}\n")
                for sec in nd.get("sections",[]):
                    md.append(f"\n## {sec.get('heading','')}")
                    if sec.get("content"): md.append(sec["content"])
                    for b in sec.get("bullets",[]): md.append(f"- {b}")
                if nd.get("key_terms"):
                    md.append("\n## Key Terms")
                    for kt in nd["key_terms"]: md.append(f"**{kt['term']}**: {kt['definition']}")
                if nd.get("exam_tips"):
                    md.append("\n## Exam Tips")
                    for t in nd["exam_tips"]: md.append(f"- ⚡ {t}")
                st.code("\n".join(md), language="markdown")

        elif not gen_notes:
            st.markdown("""
            <div class="sm-empty">
                <div class="sm-empty-icon">📓</div>
                <h4>Configure and generate</h4>
                <p>Set your topic, style, and detail level, then click Generate Notes.</p>
            </div>""", unsafe_allow_html=True)