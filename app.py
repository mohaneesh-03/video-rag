import os
import sys
import streamlit as st
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import analyze_transcript, analyze_transcript_from_index
from core.vector_store import upload_vectors
from core.rag_engine import ConversationalRAG

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Video Intelligence RAG",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom Sober Theme (Linear Slate Dark) ──────────────────────────────────
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    :root {
        --bg-main: #0e0f12;
        --surface-card: #16171d;
        --surface-card-hover: #1c1d25;
        --border-color: #232530;
        --text-primary: #f3f4f6;
        --text-secondary: #9ca3af;
        --accent-indigo: #6366f1;
        --accent-indigo-hover: #4f46e5;
        --badge-bg: #1e2029;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background-color: var(--bg-main) !important;
        color: var(--text-primary) !important;
    }

    .stApp {
        background-color: var(--bg-main) !important;
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #121318 !important;
        border-right: 1px solid var(--border-color) !important;
    }

    /* Cards */
    .metric-card {
        background-color: var(--surface-card);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }

    .metric-card-header {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-secondary);
        margin-bottom: 8px;
    }

    .metric-card-content {
        font-size: 0.95rem;
        line-height: 1.6;
        color: var(--text-primary);
        white-space: pre-wrap;
    }

    /* Hero Banner */
    .title-banner {
        background: linear-gradient(180deg, #181922 0%, var(--surface-card) 100%);
        border: 1px solid var(--border-color);
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 24px;
    }

    .title-banner h2 {
        margin: 0;
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary);
    }

    .title-banner p {
        margin: 6px 0 0 0;
        font-size: 0.85rem;
        color: var(--text-secondary);
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        font-size: 0.75rem;
        font-weight: 500;
        background-color: var(--badge-bg);
        border: 1px solid var(--border-color);
        border-radius: 4px;
        color: var(--text-secondary);
        margin-right: 6px;
    }

    /* Buttons */
    .stButton > button {
        background-color: var(--accent-indigo) !important;
        color: #ffffff !important;
        border: 1px solid var(--accent-indigo) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.15s ease-in-out !important;
    }

    .stButton > button:hover {
        background-color: var(--accent-indigo-hover) !important;
        border-color: var(--accent-indigo-hover) !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 1px solid var(--border-color);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 16px;
        color: var(--text-secondary);
    }

    .stTabs [aria-selected="true"] {
        color: var(--text-primary) !important;
        border-bottom: 2px solid var(--accent-indigo) !important;
    }

    /* Chat Messages */
    .stChatMessage {
        background-color: var(--surface-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        margin-bottom: 10px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Initialize Session State ────────────────────────────────────────────────
if "pipeline_done" not in st.session_state:
    st.session_state.pipeline_done = False

if "result" not in st.session_state:
    st.session_state.result = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = None


# ─── Sidebar Controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎙️ Video Intelligence")
    st.caption("AI-Powered Transcription, Analysis & RAG")
    st.divider()

    video_url = st.text_input(
        "YouTube Video URL",
        value="",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste any publicly accessible YouTube URL.",
    )

    top_k = st.slider(
        "Retrieval Depth (Chunks)",
        min_value=2,
        max_value=6,
        value=3,
        help="Number of transcript chunks retrieved from Qdrant per question.",
    )

    analyze_button = st.button("🚀 Analyze Video", use_container_width=True)

    st.divider()
    st.markdown("#### Pipeline Tech Stack")
    st.markdown(
        """
        - **Transcription:** Groq Whisper Large
        - **Vector DB:** Qdrant Cloud
        - **Embeddings:** MiniLM-L6-v2
        - **RAG LLM:** Groq GPT-OSS-120B
        - **Orchestration:** LangChain LCEL
        """
    )


# ─── Pipeline Execution ──────────────────────────────────────────────────────
if analyze_button:
    if not video_url.strip():
        st.warning("⚠️ Please enter a YouTube video URL first.")
    else:
        progress_status = st.status("Initiating video processing...", expanded=True)

    try:
        # Step 1: Download & convert audio
        progress_status.write("📥 Downloading and converting audio chunks...")
        chunks = process_input(video_url.strip())
        progress_status.write(f"✓ Audio split into {len(chunks)} chunk(s).")

        # Step 2: Transcription
        progress_status.write("🎙️ Transcribing with Groq Whisper...")
        transcription = transcribe_all(chunks, translate=True)
        progress_status.write("✓ Transcription complete.")

        # Step 3: Index Vectors First in Qdrant (Sentence-Window Context Expansion)
        progress_status.write("🚀 Indexing full transcript into Qdrant Vector Store (sentence-window chunking)...")
        upload_vectors(transcription)
        progress_status.write("✓ Vectors indexed with sentence-window expansion.")

        # Step 4: RAG-Driven Extraction (Bypasses 8k Context Ceiling)
        progress_status.write("📋 Extracting title, summary, action items & decisions (RAG-driven evidence synthesis)...")
        analysis = analyze_transcript_from_index(fallback_transcript=transcription)
        title = analysis["title"]
        summary = analysis["summary"]
        action_items = analysis["action_items"]
        decisions = analysis["key_decisions"]
        questions = analysis["open_questions"]
        progress_status.write("✓ Length-agnostic analysis and insights generated.")

        # Save to session state
        st.session_state.result = {
            "title": title,
            "source": video_url.strip(),
            "transcript": transcription,
            "summary": summary,
            "action_items": action_items,
            "decisions": decisions,
            "questions": questions,
        }
        st.session_state.rag_engine = ConversationalRAG(top_k=top_k)
        st.session_state.chat_history = []
        st.session_state.pipeline_done = True

        progress_status.update(label="✅ Analysis complete!", state="complete", expanded=False)
        st.rerun()

    except Exception as e:
        progress_status.update(label=f"❌ Processing Error: {e}", state="error", expanded=True)
        st.error(f"Error details: {e}")


# ─── Main Interface ──────────────────────────────────────────────────────────
if st.session_state.pipeline_done and st.session_state.result:
    res = st.session_state.result

    # Title Banner
    st.markdown(
        f"""
        <div class="title-banner">
            <span class="badge">PROCESSED VIDEO</span>
            <h2>{res['title']}</h2>
            <p>Source: {res['source']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs: Insights vs Full Transcript
    tab_insights, tab_transcript = st.tabs(["📊 Executive Summary & Insights", "📝 Full Transcript"])

    with tab_insights:
        # Executive Summary
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-card-header">Executive Summary</div>
                <div class="metric-card-content">{res['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Action Items, Decisions, Questions Grid
        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-header">✅ Action Items</div>
                    <div class="metric-card-content">{res['action_items']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-header">🔑 Key Decisions</div>
                    <div class="metric-card-content">{res['decisions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-header">❓ Open Questions</div>
                    <div class="metric-card-content">{res['questions']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with tab_transcript:
        st.markdown(f"**Word Count:** ~{len(res['transcript'].split())} words | **Characters:** {len(res['transcript'])}")
        st.text_area(
            "Complete Transcript",
            value=res["transcript"],
            height=320,
            disabled=True,
            label_visibility="collapsed",
        )

    st.divider()

    # ─── Conversational RAG Chatbot Section ──────────────────────────────────
    col_chat_title, col_chat_clear = st.columns([5, 1])
    with col_chat_title:
        st.markdown("### 💬 Conversational Video Chatbot")
        st.caption("Ask questions about specific moments, speakers, or topics discussed in this video.")
    with col_chat_clear:
        if st.button("🧹 Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            if st.session_state.rag_engine:
                st.session_state.rag_engine.clear_history()
            st.rerun()

    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "query_used" in message and message["query_used"] != message.get("original_question"):
                with st.expander("🔍 Search Query Used", expanded=False):
                    st.caption(message["query_used"])

    # Chat Input
    if user_prompt := st.chat_input("Ask a question about the video transcript..."):
        # Display user message
        st.chat_message("user").markdown(user_prompt)

        # Convert state history to LangChain BaseMessage objects
        external_history = []
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                external_history.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                external_history.append(AIMessage(content=msg["content"]))

        # Execute Conversational RAG
        with st.chat_message("assistant"):
            with st.spinner("Searching transcript and formulating answer..."):
                rag_engine = st.session_state.rag_engine
                if rag_engine is None:
                    rag_engine = ConversationalRAG(top_k=top_k)
                    st.session_state.rag_engine = rag_engine

                response = rag_engine.ask(user_prompt, external_history=external_history)
                answer = response["answer"]
                query_used = response.get("query_used", user_prompt)

                st.markdown(answer)
                if query_used != user_prompt:
                    with st.expander("🔍 Search Query Used", expanded=False):
                        st.caption(query_used)

        # Record in chat history
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_prompt,
            }
        )
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": answer,
                "original_question": user_prompt,
                "query_used": query_used,
            }
        )

else:
    # Empty State (When no video has been analyzed yet)
    st.markdown(
        """
        <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; padding: 80px 20px; text-align: center;">
            <div style="font-size: 3rem; margin-bottom: 16px;">🎙️</div>
            <h2 style="font-size: 1.6rem; font-weight: 700; margin-bottom: 8px;">Ready to Analyze Your Video</h2>
            <p style="color: var(--text-secondary); max-width: 480px; font-size: 0.95rem; line-height: 1.6;">
                Paste a YouTube URL in the left sidebar and click <strong>Analyze Video</strong>. The pipeline will transcribe the audio, extract key decisions, and initialize an interactive RAG chatbot.
            </p>
            <div style="margin-top: 24px; display: flex; gap: 8px;">
                <span class="badge">Whisper Transcription</span>
                <span class="badge">Summary & Action Items</span>
                <span class="badge">Qdrant Cloud RAG</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
