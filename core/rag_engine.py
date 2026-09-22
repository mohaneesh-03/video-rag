from groq import Groq
from dotenv import load_dotenv
import os
from core.vector_store import search

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")


groq_client = Groq(api_key=groq_api_key)


def ask_llm(question: str, context: str):
    prompt = f"""
    Answer the question based on the context below:
    
    Question: {question}
    Context: {context}

    If answer is not in context, say "I don't know."
    """
    response = groq_client.chat.completions.create(
        model = "openai/gpt-oss-120b",
        messages = [{
            "role": "user",
            "content": prompt
        }]
    )
    return response.choices[0].message.content

def ask_question(question:str):
    search(question)