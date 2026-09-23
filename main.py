import sys
from core.extractor import analyze_transcript
from core.vector_store import upload_vectors
from core.rag_engine import ConversationalRAG

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=4JKOGT8HH1Q"

print("\n" + "=" * 60)
print(f"🎬 Processing Video: {source}")
print("=" * 60)

chunks = process_input(source)
print(f"Audio split into {len(chunks)} chunks.")

transcription = transcribe_all(chunks, translate=True)
print("\n" + "=" * 60)
print("📝 TRANSCRIPTION COMPLETED")
print("=" * 60)
print(transcription[:500] + "...\n(truncated for display)")

print("\n" + "=" * 60)
print("📋 GENERATING INSIGHTS (SINGLE PASS)")
print("=" * 60)
analysis = analyze_transcript(transcription)
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

# Index transcript into Qdrant
print("\n" + "=" * 60)
print("🚀 Indexing transcript into Qdrant Vector Store...")
print("=" * 60)
upload_vectors(transcription)

# Launch Conversational RAG Chatbot
bot = ConversationalRAG(top_k=3)
bot.cli_chat()