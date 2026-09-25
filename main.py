import sys
from dotenv import load_dotenv

load_dotenv()

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all
from core.extractor import analyze_transcript_from_index
from core.vector_store import upload_vectors
from core.rag_engine import ConversationalRAG


def main():
    # ─── Dynamic Input (CLI Argument or Interactive Prompt) ───────────────────
    if len(sys.argv) > 1 and sys.argv[1].strip():
        source = sys.argv[1].strip()
    else:
        source = input("\nEnter YouTube Video URL: ").strip()

    while not source:
        source = input("URL cannot be empty. Please enter YouTube Video URL: ").strip()

    print("\n" + "=" * 60)
    print(f"🎬 Processing Video: {source}")
    print("=" * 60)

    # Step 1: Download & Chunk Audio
    chunks = process_input(source)
    print(f"Audio split into {len(chunks)} chunks.")

    # Step 2: Whisper Transcription
    transcription = transcribe_all(chunks, translate=True)
    print("\n" + "=" * 60)
    print("📝 TRANSCRIPTION COMPLETED")
    print("=" * 60)
    print(transcription[:400] + "...\n(truncated for terminal display)")

    # Step 3: Index Vectors First in Qdrant (Sentence-Window Context Expansion)
    print("\n" + "=" * 60)
    print("🚀 Indexing transcript into Qdrant Vector Store (Sentence-Window)...")
    print("=" * 60)
    upload_vectors(transcription)

    # Step 4: Length-Agnostic RAG-Driven Extraction
    print("\n" + "=" * 60)
    print("📋 GENERATING INSIGHTS (RAG-Driven Evidence Extraction)")
    print("=" * 60)
    analysis = analyze_transcript_from_index(fallback_transcript=transcription)
    title = analysis["title"]
    summary = analysis["summary"]
    action_items = analysis["action_items"]
    decisions = analysis["key_decisions"]
    questions = analysis["open_questions"]

    print(f"\n📌 TITLE: {title}")
    print("\n📋 SUMMARY")
    print("-" * 60)
    print(summary)

    print("\n" + "=" * 60)
    print("✅ ACTION ITEMS")
    print("=" * 60)
    print(action_items)

    print("\n" + "=" * 60)
    print("🔑 KEY DECISIONS")
    print("=" * 60)
    print(decisions)

    print("\n" + "=" * 60)
    print("❓ OPEN QUESTIONS")
    print("=" * 60)
    print(questions)

    # Step 5: Launch Conversational RAG Chatbot
    bot = ConversationalRAG(top_k=3)
    bot.cli_chat()


if __name__ == "__main__":
    main()