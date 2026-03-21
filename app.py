import streamlit as st
from sentence_transformers import SentenceTransformer
from chromadb import PersistentClient
import ollama
from pypdf import PdfReader
import uuid
import re

# ---------- CONFIG ----------
st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
html, body {
    font-family: 'Inter', sans-serif;
    background-color: #0b1220;
}

.block-container {
    max-width: 900px;
    margin: auto;
}

.header {
    padding: 16px 0;
    border-bottom: 1px solid #1f2937;
    margin-bottom: 20px;
}

.stChatMessage {
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 10px;
    max-width: 80%;
}

[data-testid="stChatMessage"]:has(div[aria-label="user"]) {
    background-color: #1f2937;
    margin-left: auto;
}

[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
    background-color: #111827;
    margin-right: auto;
}

textarea {
    border-radius: 8px !important;
}

.source-card {
    padding:10px;
    border-radius:10px;
    background:#1f2937;
    margin-bottom:10px;
    font-size:14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD ----------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)

@st.cache_resource
def load_db():
    client = PersistentClient(path="chroma_db")
    return client.get_or_create_collection("docs")

model = load_model()
collection = load_db()

# ---------- CHUNKING ----------
def chunk_text(text, chunk_size=800, overlap=150):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i+chunk_size].strip()
        if len(chunk) > 50:
            chunks.append(chunk)
    return chunks

# ---------- PDF PROCESS ----------
def process_pdfs(files):
    all_chunks, all_embeddings, all_ids, all_metadata = [], [], [], []

    for file in files:
        file.seek(0)
        reader = PdfReader(file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        # 🔥 DEBUG START
        print("\n\n----- DEBUG TEXT START -----\n")
        print(text[:1000])
        print("\n----- DEBUG TEXT END -----\n\n")
        # 🔥 DEBUG END

        chunks = chunk_text(text)

        embeddings = model.encode(chunks)

        ids = [str(uuid.uuid4()) for _ in chunks]
        metadata = [{"source": file.name}] * len(chunks)

        all_chunks.extend(chunks)
        all_embeddings.extend(embeddings)
        all_ids.extend(ids)
        all_metadata.extend(metadata)

    if all_chunks:
        collection.add(
            documents=all_chunks,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata
        )

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("AI Assistant")

    uploaded_files = st.file_uploader(
        "Upload PDFs",
        type="pdf",
        accept_multiple_files=True
    )

    if uploaded_files:
        st.markdown("### Uploaded Files")
        for file in uploaded_files:
            st.write("•", file.name)

    if uploaded_files and st.button("Process Documents"):
        process_pdfs(uploaded_files)
        st.success("Documents processed")

    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Model: Llama 3.1")
    st.caption("Embeddings: MiniLM")

# ---------- HEADER ----------
st.markdown("""
<div class="header">
    <h2 style="margin:0;">AI Study Assistant</h2>
    <p style="color:#9ca3af; margin:0;">
        Ask questions from your academic PDFs
    </p>
</div>
""", unsafe_allow_html=True)

# ---------- TABS ----------
tab1, tab2 = st.tabs(["💬 Chat", "📄 Documents"])

# ---------- STATE ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- CHAT ----------
with tab1:

    if not st.session_state.messages:
        st.markdown("""
    <div style="
        padding:20px;
        border-radius:10px;
        background:#111827;
        color:#9ca3af;
        text-align:center;
        margin-bottom:20px;
    ">
        Upload documents and start asking questions.
    </div>
    """, unsafe_allow_html=True)

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about your documents...")

    if question:

        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):

                module_match = re.search(r"module\s*(\d+)", question.lower())
                module_number = module_match.group(1) if module_match else None

                # ---------- RETRIEVAL ----------
                if module_number:
                    search_query = f"module {module_number} data structures topics"
                    n_results = 10
                else:
                    search_query = question
                    n_results = 6

                embedding = model.encode([search_query])[0]

                results = collection.query(
                    query_embeddings=[embedding],
                    n_results=n_results
                )

                documents = results["documents"][0]
                metadatas = results["metadatas"][0]

                documents = [d for d in documents if d.strip()]

                if module_number:
                    filtered = [
                        d for d in documents
                        if f"module {module_number}" in d.lower()
                        or f"module-{module_number}" in d.lower()
                    ]
                    context = "\n".join(filtered)
                else:
                    context = "\n".join(documents)

                # ---------- PROMPT ----------
                prompt = f"""
You are an AI assistant.

STRICT RULES:

1. Answer using ONLY the provided context.
2. If the answer is present in the context → answer clearly.
3. If the answer is NOT present → say:
   "I could not find this information in the document."

4. DO NOT use outside knowledge.
5. DO NOT add extra explanations beyond the context.

Context:
{context}

Question:
{question}
"""

                response = ollama.chat(
                    model="llama3.1",
                    messages=[{"role": "user", "content": prompt}],
                    stream=True
                )

                full = ""
                placeholder = st.empty()

                for chunk in response:
                    content = chunk["message"]["content"]
                    full += content
                    placeholder.markdown(full)

                answer = full

                # ---------- SOURCES ----------
                if "general knowledge" not in answer.lower() and context.strip():
                    st.markdown("### Sources")

                    for i, doc in enumerate(documents):
                        preview = doc[:200].replace("\n", " ") + "..."
                        source_name = metadatas[i]["source"]

                        st.markdown(f"""
                        <div class="source-card">
                        <b>{source_name}</b><br>
                        {preview}
                        </div>
                        """, unsafe_allow_html=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

# ---------- DOCUMENT TAB ----------
with tab2:
    st.markdown("### Uploaded Documents")

    if uploaded_files:
        for file in uploaded_files:
            st.markdown(f"- {file.name}")
    else:
        st.info("No documents uploaded yet.")