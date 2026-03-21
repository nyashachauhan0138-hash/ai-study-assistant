# AI Study Assistant

An AI-powered document assistant that enables users to interact with PDFs using **Retrieval-Augmented Generation (RAG)**.

---

## 🚀 Features

* 📄 Upload and query multiple PDFs
* 🔍 Semantic search using embeddings
* 🧠 Context-aware answers from documents
* ⚖️ Smart fallback to general knowledge when needed
* 💬 Chat-based interface (Streamlit)
* 📌 Source attribution for transparency

---

## 🧠 How It Works

1. PDFs are parsed and split into chunks
2. Chunks are converted into embeddings using Sentence Transformers
3. Stored in a vector database (ChromaDB)
4. User query → converted to embedding
5. Relevant chunks retrieved
6. LLM (Llama 3.1 via Ollama) generates answer using context

---

## 🛠 Tech Stack

* **Frontend**: Streamlit
* **LLM**: Llama 3.1 (Ollama)
* **Embeddings**: Sentence Transformers (MiniLM)
* **Vector DB**: ChromaDB
* **Language**: Python

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## ⚠️ Notes

* Upload your own PDFs to query
* Local LLM required (Ollama installed)
* No external APIs used

---

## 🎯 Future Improvements

* Better document filtering
* Multi-file context separation
* UI enhancements
* Deployment (Streamlit Cloud / Docker)

---

## 👤 Author

Nyasha Chauhan
AI/ML Engineer | Building Intelligent Systems
