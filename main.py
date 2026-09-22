import sys
from core.summarize import generate_title
from core.summarize import summarize
from core.extractor import extract_actions_items, extract_key_decisions, extract_questions

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=4JKOGT8HH1Q"

chunks = process_input(source)
print(chunks)

transcription = transcribe_all(chunks, translate=True)
print(transcription)

title = generate_title(transcription)
summary = summarize(transcription)

print("\n" + "=" * 60)
print(f"📌 TITLE: {title}")
print("=" * 60)
print("\n📋 SUMMARY")
print("-" * 60)
print(summary)



action_items = extract_actions_items(transcription)
decisions = extract_key_decisions(transcription)
questions = extract_questions(transcription)

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