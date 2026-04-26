import streamlit as st
import requests

BACKEND_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="Document Intelligence RAG",
    page_icon="📄",
    layout="wide"
)

# ---- Session State Init ----
if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_loaded" not in st.session_state:
    st.session_state.pdf_loaded = False

# ---- Sidebar ----
with st.sidebar:
    st.title("⚙️ Settings")

    st.subheader("📂 Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

    if uploaded_file and not st.session_state.pdf_loaded:
        with st.spinner("Indexing document..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    timeout=120
                )
                if response.status_code == 200:
                    st.success(f"✅ '{uploaded_file.name}' indexed!")
                    st.session_state.pdf_loaded = True
                    st.session_state.messages = []  # reset chat for new doc
                else:
                    st.error(f"Upload failed: {response.text}")
            except Exception as e:
                st.error(f"Connection error: {e}")

    st.divider()
    top_k = st.slider("Retrieval depth (top_k)", 3, 10, 5,
                      help="Number of document chunks retrieved per query")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.caption("**How it works:**")
    st.caption("🔍 Hybrid search: FAISS + BM25")
    st.caption("🏆 Cross-encoder reranking")
    st.caption("💬 Multi-turn conversation")
    st.caption("🤖 Gemini 2.0 Flash")

# ---- Main Chat UI ----
st.title("📄 Document Intelligence RAG")
st.caption("Ask questions about your uploaded research paper or document.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander(f"📚 View {len(message['sources'])} retrieved sources"):
                for i, src in enumerate(message["sources"], 1):
                    st.markdown(f"**Chunk {i}** — Relevance score: `{src['score']:.3f}`")
                    st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                    st.divider()

# Chat Input
if prompt := st.chat_input("Ask a question about the document..."):

    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build chat history for backend (exclude last user msg, it's the current query)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages[:-1]
    ]

    # Query backend
    with st.chat_message("assistant"):
        with st.spinner("Searching and generating answer..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={
                        "query": prompt,
                        "top_k": top_k,
                        "chat_history": history
                    },
                    timeout=120
                )

                if response.status_code == 200:
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]

                    st.markdown(answer)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                    with st.expander(f"📚 View {len(sources)} retrieved sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}** — Relevance score: `{src['score']:.3f}`")
                            st.text(src["text"][:500] + ("..." if len(src["text"]) > 500 else ""))
                            st.divider()
                else:
                    st.error(f"Backend error {response.status_code}: {response.text}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot connect to backend. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
