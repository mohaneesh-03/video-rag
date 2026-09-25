import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

groq_client = Groq(api_key=groq_api_key)

import time


def transcribe(file_path: str, translate: bool = False) -> str:
    for attempt in range(5):
        try:
            if not translate:
                with open(file_path, "rb") as file:
                    transcription = groq_client.audio.transcriptions.create(
                        file=file,
                        model="whisper-large-v3",
                        response_format="verbose_json",
                        language="en",
                    )
                    return transcription.text
            else:
                with open(file_path, "rb") as file:
                    translation = groq_client.audio.translations.create(
                        file=file,
                        model="whisper-large-v3",
                        response_format="json",
                    )
                    return translation.text
        except Exception as e:
            if attempt < 4:
                wait_sec = (2 ** attempt) * 2
                print(f"⚠️ Groq transcription attempt {attempt+1} failed: {e}. Retrying in {wait_sec}s...")
                time.sleep(wait_sec)
            else:
                raise e


def transcribe_all(chunks: list, translate: bool = False) -> str:
    full_transcript = ""

    for i, chunk in enumerate(chunks):
        print(f"Transcribing {i+1}th chunk...")
        text = transcribe(chunk, translate=translate)
        full_transcript += text + " "
        time.sleep(1)  # small pause to respect rate limits

    print("Transcription Completed")

    return full_transcript