"""
streamlit_app.py
Simple chat-style frontend for the RAG PDF Q&A Assistant.
Talks to the FastAPI backend over HTTP.

Run the backend first:
    uvicorn app.main:app --reload

Then run this:
    streamlit run ui/streamlit_app.py
"""

import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000"

st.set_page_config(page_title="RAG PDF Q&A Assistant", page_icon="📄")
st.title("📄 RAG PDF Q&A Assistant")
st.caption("Upload a PDF and ask questions grounded in its actual content.")

if "indexed" not in st.session_state:
    st.session_state.indexed = False
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Upload section ---
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

if uploaded_file is not None and not st.session_state.indexed:
    with st.spinner("Reading and indexing your document..."):
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
        try:
            response = requests.post(f"{API_URL}/upload", files=files, timeout=120)
            response.raise_for_status()
            data = response.json()
            st.session_state.indexed = True
            st.success(f"Indexed {data['chunks_indexed']} chunks from {data['filename']}.")
        except requests.exceptions.RequestException as e:
            st.error(f"Upload failed: {e}")

# --- Chat section ---
if st.session_state.indexed:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question about the document...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = requests.post(
                        f"{API_URL}/ask", json={"query": question, "top_k": 3}, timeout=60
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]
                    st.write(answer)

                    if data.get("sources"):
                        with st.expander("Sources"):
                            for src in data["sources"]:
                                st.markdown(f"**Page {src['page']}:** {src['excerpt']}...")

                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except requests.exceptions.RequestException as e:
                    st.error(f"Request failed: {e}")
else:
    st.info("Upload a PDF above to get started.")
