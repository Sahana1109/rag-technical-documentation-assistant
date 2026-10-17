"""
Streamlit Web Interface for RAG Technical Documentation Assistant
Communicates with the FastAPI backend REST API.
"""
import os
import requests
import streamlit as st

# Configure backend API URL
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="FastAPI Documentation Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #009688;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #607d8b;
        margin-bottom: 1.5rem;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        background-color: #e0f2f1;
        color: #00796b;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
    }
    .citation-card {
        padding: 10px;
        background-color: #f8f9fa;
        border-left: 4px solid #009688;
        border-radius: 4px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------------------
# Sidebar: System Diagnostics, Document Catalog, and Ingestion
# ------------------------------------------------------------------------------

with st.sidebar:
    st.header("⚡ System Status")
    try:
        health_res = requests.get(f"{API_URL}/health", timeout=3)
        if health_res.status_code == 200:
            health = health_res.json()
            st.success(f"Backend: {health['status'].upper()}")
            st.markdown(f"**Indexed Chunks:** `{health['total_indexed_chunks']}`")
            st.markdown(f"**LLM Provider:** `{health['llm_provider']}`")
            st.markdown(f"**Embedding Model:** `{health['embedding_model']}`")
        else:
            st.warning("Backend degraded")
    except Exception:
        st.error(f"Cannot connect to API at `{API_URL}`. Ensure FastAPI is running on port 8000.")

    st.divider()

    st.header("📚 Document Catalog")
    try:
        docs_res = requests.get(f"{API_URL}/documents", timeout=3)
        if docs_res.status_code == 200:
            catalog = docs_res.json()
            st.caption(f"Total: {catalog['total_documents']} docs ({catalog['total_chunks']} chunks)")
            for doc in catalog["documents"]:
                with st.expander(f"📄 {doc['document_title']}"):
                    st.write(f"**File:** `{doc['source_filename']}`")
                    st.write(f"**Chunks:** {doc['chunk_count']}")
                    if doc["source_url"]:
                        st.markdown(f"[Official Documentation]({doc['source_url']})")
    except Exception as e:
        st.caption("Unable to load document catalog.")

    st.divider()

    st.header("📥 Ingest Document")
    with st.form("ingest_file_form"):
        uploaded_file = st.file_uploader("Upload .md, .txt, or .html", type=["md", "txt", "html"])
        url_input = st.text_input("Or Ingest Documentation URL", placeholder="https://fastapi.tiangolo.com/...")
        submit_ingest = st.form_submit_button("Ingest Document")

        if submit_ingest:
            if not uploaded_file and not url_input:
                st.error("Please provide a file or URL.")
            else:
                try:
                    files = None
                    data = None
                    if uploaded_file:
                        files = [("files", (uploaded_file.name, uploaded_file.getvalue(), "text/markdown"))]
                    if url_input:
                        data = {"url": url_input.strip()}

                    res = requests.post(f"{API_URL}/ingest", files=files, data=data, timeout=15)
                    if res.status_code == 200:
                        ingest_data = res.json()
                        st.success(f"Indexed {ingest_data['documents_ingested']} doc(s) ({ingest_data['chunks_created']} chunks)!")
                        st.rerun()
                    else:
                        st.error(f"Ingestion failed: {res.json().get('detail', res.text)}")
                except Exception as ex:
                    st.error(f"Error during ingestion: {ex}")


# ------------------------------------------------------------------------------
# Main Area: Query Assistant and Answer Generation
# ------------------------------------------------------------------------------

st.markdown('<div class="main-title">FastAPI Documentation Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Self-Corrective RAG powered by LangGraph, ChromaDB, and Sentence Transformers</div>', unsafe_allow_html=True)

# Quick example question buttons
st.markdown("**Quick Questions:**")
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("🔌 How do dependencies work?"):
        st.session_state["query_input"] = "How do dependencies work in FastAPI?"
with col2:
    if st.button("🛣️ How to use APIRouter for routing?"):
        st.session_state["query_input"] = "How to use APIRouter for modular routing in FastAPI?"
with col3:
    if st.button("🛡️ How does Pydantic request validation work?"):
        st.session_state["query_input"] = "How does Pydantic request validation and error handling work in FastAPI?"

# Query input
default_q = st.session_state.get("query_input", "")
query_text = st.text_input("Enter your technical question:", value=default_q, placeholder="e.g. How do yield dependencies work in FastAPI?")

if st.button("Ask Assistant", type="primary"):
    if not query_text.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Analyzing query, searching documentation, and verifying grounding..."):
            try:
                res = requests.post(f"{API_URL}/query", json={"question": query_text.strip()}, timeout=20)
                if res.status_code == 200:
                    data = res.json()
                    st.session_state["latest_query_response"] = data
                    st.session_state["latest_question"] = query_text.strip()
                else:
                    st.error(f"Query Error ({res.status_code}): {res.json().get('detail', res.text)}")
            except Exception as e:
                st.error(f"Failed to communicate with API: {e}")

# Display latest query response
if "latest_query_response" in st.session_state:
    data = st.session_state["latest_query_response"]
    q_asked = st.session_state.get("latest_question", "")

    st.divider()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Query Classification", data.get("query_type", "conceptual").capitalize())
    with m2:
        st.metric("Retrieval Retries", data.get("retry_count", 0))
    with m3:
        status_label = data.get("grounding_status", "not_checked").replace("_", " ").capitalize()
        st.metric("Grounding Status", status_label)
    with m4:
        st.metric("Context Sufficient", "Yes" if data.get("sufficient_context", True) else "No")

    # Answer Markdown
    st.markdown("### Answer")
    st.markdown(data.get("answer", ""))

    # Citations
    citations = data.get("citations", [])
    if citations:
        st.markdown("### Verified Citations")
        for i, c in enumerate(citations):
            title = c.get("document_title", "Document")
            section = c.get("section", "Section")
            url = c.get("source_url")
            link_html = f'<a href="{url}" target="_blank">View Official Doc ↗</a>' if url else "Local Corpus"

            st.markdown(
                f'<div class="citation-card">'
                f'<strong>[{i+1}] {title}</strong> — <em>Section: {section}</em><br>'
                f'<small>Source: {c.get("source_filename")} | {link_html}</small>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Relevant Sources Expander
    sources = data.get("relevant_sources", [])
    if sources:
        with st.expander(f"🔍 Inspect {len(sources)} Retrieved Context Chunks"):
            for s in sources:
                st.markdown(f"**{s.get('document_title')}** (Section: *{s.get('section')}*) — Relevance Score: `{s.get('score', 'N/A')}`")
                st.code(s.get("snippet", ""), language="markdown")

    # Feedback Section
    st.divider()
    st.markdown("### Was this answer helpful?")
    fb_col1, fb_col2, _ = st.columns([1, 1, 4])
    with fb_col1:
        if st.button("👍 Thumbs Up"):
            try:
                f_res = requests.post(f"{API_URL}/feedback", json={
                    "rating": "thumbs_up",
                    "question": q_asked,
                    "comment": "User clicked Thumbs Up"
                })
                if f_res.status_code == 200:
                    st.success("Thank you for your feedback!")
            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")
    with fb_col2:
        if st.button("👎 Thumbs Down"):
            try:
                f_res = requests.post(f"{API_URL}/feedback", json={
                    "rating": "thumbs_down",
                    "question": q_asked,
                    "comment": "User clicked Thumbs Down"
                })
                if f_res.status_code == 200:
                    st.info("Feedback noted. We will improve our documentation retrieval.")
            except Exception as e:
                st.error(f"Failed to submit feedback: {e}")
