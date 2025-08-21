# app.py
# Streamlit + Ollama local RAG chatbot with owner metadata extraction, guardrails, and persistent chat UI.

import re, os, tempfile
import streamlit as st
from typing import List, Tuple, Dict

# Vector DB + loaders
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import (
    PyPDFLoader, TextLoader, Docx2txtLoader
)
# We'll read .md via TextLoader to avoid heavy unstructured deps.

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Ollama (NEW package to avoid deprecation warnings)
from langchain_ollama import OllamaEmbeddings, OllamaLLM

# -----------------------------
# Config
# -----------------------------
DEFAULT_MODEL = "llama3.2:3b"
EMBED_MODEL   = "nomic-embed-text"
CHROMA_DIR    = "chroma_db"
K_RETRIEVE    = 4

st.set_page_config(page_title="Shalini’s RAG Chatbot", page_icon="🤖", layout="wide")

# -----------------------------
# Simple CSS to make it pretty
# -----------------------------
st.markdown("""
<style>
/* app padding */
.block-container {padding-top: 2rem; padding-bottom: 2rem;}
/* chat bubbles */
.chat-bubble-user, .chat-bubble-bot {
  padding: 12px 14px; border-radius: 14px; margin: 6px 0; line-height: 1.45;
  border: 1px solid rgba(0,0,0,0.06);
  box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.chat-bubble-user { background: #f4f8ff; }
.chat-bubble-bot  { background: #f8f9fb; }
.small { font-size: 0.9rem; color: #666; }
.owner-pill { background:#eefaf1; color:#16794c; border:1px solid #cdeede;
              padding:6px 10px; border-radius: 999px; display:inline-block; }
hr { border: none; border-top: 1px solid #eee; margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# Helper: extract owner metadata
# -----------------------------
NAME_HINTS  = r"(?:name|candidate|owner|profile)\s*[:\-]?\s*([A-Z][A-Za-z\.\s\-]{1,60})"
EMAIL_REGEX = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
PHONE_REGEX = r"(?:\+?\d[\d\s\-().]{7,}\d)"

LINK_REGEX  = r"(?:https?://[^\s)]+|(?:github|linkedin)\.com/[^\s)]+)"

def extract_owner_metadata(full_text: str) -> Dict[str, str]:
    """
    Heuristic extraction from the first ~2 pages / first ~5000 chars.
    """
    focus = full_text[:5000] if full_text else ""
    meta = {}

    # email
    emails = re.findall(EMAIL_REGEX, focus, flags=re.IGNORECASE)
    if emails:
        meta["email"] = emails[0]

    # phone
    phones = re.findall(PHONE_REGEX, focus)
    if phones:
        # choose a reasonable-looking one (trim spaces)
        meta["phone"] = re.sub(r"\s+", " ", phones[0]).strip()

    # links (github/linkedin/portfolio)
    links = re.findall(LINK_REGEX, focus, flags=re.IGNORECASE)
    if links:
        # keep distinct + readable
        uniq = []
        for l in links:
            if l not in uniq:
                uniq.append(l)
        meta["links"] = ", ".join(uniq[:3])

    # try explicit "Name:"-like hints
    m = re.search(NAME_HINTS, focus, flags=re.IGNORECASE)
    if m:
        meta["name"] = m.group(1).strip()

    # otherwise guess first prominent line (resume headers often start with name)
    if "name" not in meta:
        # first 5 lines heuristic: pick a line of letters with <= 5 words and Title Case
        lines = [ln.strip() for ln in focus.splitlines()[:20] if ln.strip()]
        for ln in lines:
            words = ln.split()
            if 1 <= len(words) <= 6 and ln[0].isupper() and re.match(r"[A-Za-z][A-Za-z\s\.\-]+$", ln):
                meta["name"] = ln
                break

    return meta

def owner_chunk_from_meta(meta: Dict[str, str]) -> str:
    if not meta:
        return ""
    parts = []
    if meta.get("name"):  parts.append(f"Name: {meta['name']}")
    if meta.get("email"): parts.append(f"Email: {meta['email']}")
    if meta.get("phone"): parts.append(f"Phone: {meta['phone']}")
    if meta.get("links"): parts.append(f"Links: {meta['links']}")
    return " | ".join(parts)

# -----------------------------
# File loading
# -----------------------------
def load_files_to_docs(uploaded_files) -> Tuple[List[Document], str, Dict[str, str]]:
    """
    Returns (all_docs, full_text, owner_meta)
    Reads PDFs with PyPDFLoader, DOCX with Docx2txtLoader, TXT/MD with TextLoader.
    """
    all_docs: List[Document] = []
    merged_texts = []

    for f in uploaded_files:
        suffix = os.path.splitext(f.name)[1].lower()
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.read())
            path = tmp.name

        if suffix == ".pdf":
            loader = PyPDFLoader(path)
        elif suffix in (".txt", ".md"):
            loader = TextLoader(path, encoding="utf-8")
        elif suffix == ".docx":
            loader = Docx2txtLoader(path)
        else:
            st.warning(f"Unsupported file skipped: {f.name}")
            continue

        docs = loader.load()
        all_docs.extend(docs)
        merged_texts.append("\n".join(d.page_content for d in docs))

    full_text = "\n".join(merged_texts)
    owner_meta = extract_owner_metadata(full_text)
    return all_docs, full_text, owner_meta

# -----------------------------
# Build guarded prompt
# -----------------------------
def build_prompt(query: str, context: str, owner_meta: Dict[str, str], chat_history:str="") -> str:
    """
    Guardrails:
    - Never invent names or details; if not present, say "not found in document".
    - If summarizing a resume/CV, begin with "This is the CV/Resume of ..."
    - Cite info only from provided context/history.
    """
    owner_line = owner_chunk_from_meta(owner_meta)
    return f"""
You are an AI assistant analyzing user-uploaded documents with retrieval-augmented context.
STRICT RULES:
- Do NOT invent names or personal details. If a name/contact is not present in the context, say: "Name not found in document."
- Prefer real details from the context over generic phrasing.
- If the document is a resume/CV and a name is available, begin summaries with: "This is the CV/Resume of <Name> ..."
- Keep answers grounded in the context. If you don't find enough info, say so briefly.

Owner (if known): {owner_line if owner_line else "Not found in document"}
Chat History (may include prior Q/A to help disambiguate):
{chat_history}

Context:
{context}

User Question:
{query}

Your Answer:
""".strip()

# -----------------------------
# Initialize (session state)
# -----------------------------
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "owner_meta" not in st.session_state:
    st.session_state.owner_meta = {}
if "messages" not in st.session_state:
    st.session_state.messages = []   # list of {"role":"user"/"assistant","content":str}
if "model_name" not in st.session_state:
    st.session_state.model_name = DEFAULT_MODEL

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    model = st.selectbox(
        "Ollama model",
        options=[DEFAULT_MODEL, "mistral:latest", "llama3.2:1b"],
        index=0
    )
    st.session_state.model_name = model

    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
    if st.button("🗑️ Reset Knowledge Base"):
        st.session_state.vectordb = None
        st.session_state.owner_meta = {}
        if os.path.isdir(CHROMA_DIR):
            try:
                # wipe local DB
                import shutil
                shutil.rmtree(CHROMA_DIR)
            except Exception:
                pass
        st.success("Knowledge base reset.")

st.title("🤖 Local RAG Chatbot — Beautiful, Accurate & Grounded")

st.markdown(
    "<span class='small'>Powered by Ollama · llama3.2:3b · nomic-embed-text · ChromaDB</span>",
    unsafe_allow_html=True
)
st.markdown("---")

# Upload & Process
col1, col2 = st.columns([2,1])
with col1:
    files = st.file_uploader(
        "Upload your documents (.pdf, .txt, .md, .docx)",
        type=["pdf","txt","md","docx"],
        accept_multiple_files=True
    )
with col2:
    process = st.button("📥 Process Documents", use_container_width=True)

if process and files:
    with st.spinner("Processing documents, extracting owner details, and building the vector index..."):
        docs, full_text, owner = load_files_to_docs(files)
        splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)
        chunks = splitter.split_documents(docs)

        # ensure owner chunk is always present in KB (high-priority)
        owner_card = owner_chunk_from_meta(owner)
        if owner_card:
            chunks.append(Document(page_content=f"OWNER_METADATA:\n{owner_card}"))

        # embeddings + vectordb (persist)
        embeddings = OllamaEmbeddings(model=EMBED_MODEL)
        vectordb = Chroma.from_documents(chunks, embedding=embeddings, persist_directory=CHROMA_DIR)

        st.session_state.vectordb = vectordb
        st.session_state.owner_meta = owner

    # Owner pill
    if st.session_state.owner_meta:
        st.markdown(
            f"<div class='owner-pill'>🧑 Document owner detected — {owner_chunk_from_meta(st.session_state.owner_meta)}</div>",
            unsafe_allow_html=True
        )
    else:
        st.info("No explicit owner details detected in the uploaded documents.")

st.markdown("---")

# -----------------------------
# Chat UI (persistent)
# -----------------------------
# Render previous messages
for msg in st.session_state.messages:
    role = "You" if msg["role"] == "user" else "AI"
    css  = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
    st.markdown(f"<div class='{css}'><b>{role}:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Input
user_text = st.chat_input("Ask about your documents… (e.g., 'Summarize the document', 'Who does this belong to?')")
if user_text:
    # append user turn
    st.session_state.messages.append({"role": "user", "content": user_text})
    st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {user_text}</div>", unsafe_allow_html=True)

    # Answer
    with st.spinner("Thinking…"):
        if st.session_state.vectordb is None:
            answer = "Please upload and process documents first."
        else:
            retriever = st.session_state.vectordb.as_retriever(search_kwargs={"k": K_RETRIEVE})
            # retrieve relevant chunks
            rel_docs = retriever.get_relevant_documents(user_text)
            context = "\n\n".join(d.page_content for d in rel_docs)

            # include a short rolling chat transcript to keep followups coherent
            history_pairs = []
            for m in st.session_state.messages[-6:]:  # last few turns
                prefix = "User" if m["role"] == "user" else "AI"
                history_pairs.append(f"{prefix}: {m['content']}")
            chat_history_str = "\n".join(history_pairs[:-1])  # exclude current user turn which is already passed as query

            prompt = build_prompt(user_text, context, st.session_state.owner_meta, chat_history=chat_history_str)

            llm = OllamaLLM(model=st.session_state.model_name)
            answer = llm.invoke(prompt)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.markdown(f"<div class='chat-bubble-bot'><b>AI:</b> {answer}</div>", unsafe_allow_html=True)
