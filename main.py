import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from utils.audio_processor import process_input
from core.transcriber import transcribe_all

source = "https://www.youtube.com/watch?v=4JKOGT8HH1Q"

chunks = process_input(source)
print(chunks)

transcription = transcribe_all(chunks, translate=True)
print(transcription)