#  Cygnus AI Local RAG Chatbot

A **local, offline Retrieval-Augmented Generation (RAG) chatbot** built with:

- **Ollama** for LLM inference (e.g., `llama3.2:3b`, `mistral:latest`)
- **Chroma** for vector embeddings and storage  
- **Streamlit** UI with chat-style bubbles and persistent chat history  
- **Multi-format document ingestion**: PDF, TXT, MD, DOCX  
- **Owner metadata extraction** (name, email, phone, links) from resumes, for accurate identity responses  
- **Smart routing**: General knowledge questions fetch from “book” docs, profile questions fetch from “resume” docs  
- **Source citations** and guardrails to prevent hallucination  

---

##  Features

| Feature                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| Model Setup                      | Local LLMs via Ollama (`llama3.2:3b` default, others selectable)           |
| Document Ingestion               | Supports `.pdf`, `.txt`, `.md`, `.docx`                                     |
| Embeddings                       | `nomic-embed-text` via Ollama                                               |
| Vector DB                        | Chroma, with persistent storage (`chroma_db`)                              |
| RAG + Context Injection          | Retrieves relevant chunks, injects owner metadata as high-priority context  |
| Routing Logic                    | Distinguishes between “general AI” vs “profile/CV” queries                  |
| UI                               | Chat-style bubbles, clear chat button, model selector, file scope control   |
| Deployment Ready                 | Cloud-friendly (Render, etc.) — environment paths handled via `CHROMA_DIR`  |

---

##  Getting Started (Local)

### 1. Clone & Set Up
```bash
git clone https://github.com/Shalini-190/cygnus.git
cd cygnus
2. Pull Ollama Models & Serve
Ensure Ollama is installed and running:

bash
Copy
Edit
ollama serve
ollama pull llama3.2:3b
ollama pull nomic-embed-text
# (Optional)
ollama pull mistral:latest
ollama pull llama3.2:1b
3. Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
4. Run the App
bash
Copy
Edit
streamlit run app.py
