import os
import json
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in environment variables")

groq_client = Groq(api_key=groq_api_key)

def transcribe(file_path:str, translate:bool = False) -> str:


    if translate == False:
        with open(file_path, "rb") as file:
            transcription = groq_client.audio.transcriptions.create(
                file = file,
                model = "whisper-large-v3",
                response_format = "verbose_json",
                language = "en"
            )

            return transcription.text
    else:
        with open(file_path, "rb") as file:

            translation = groq_client.audio.translations.create(
                file = file,
                model = "whisper-large-v3",
                response_format = "json"
            )

            return translation.text
    
def transcribe_all(chunks: list, translate:bool = False) -> str:
    full_transcript = ""

    for i,chunk in enumerate(chunks):
        print(f"Transcribing {i+1}th chunk")
        text = transcribe(chunk, translate=translate)
        full_transcript += text + " "

    print("Transcription Completed")

    return full_transcript