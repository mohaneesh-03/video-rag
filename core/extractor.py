from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

import os
from time import sleep

def get_llm():
    sleep(10)
    return ChatGroq(model = "openai/gpt-oss-120b", api_key = os.getenv("GROQ_API_KEY"), temperature=0.3)

def build_chain(sys_prompt: str):

    llm = get_llm()
    return (RunnablePassthrough() | RunnableLambda(lambda x:{"text" : x}) | 
        ChatPromptTemplate.from_messages([
            ("system", sys_prompt),
            ("human", "{text}")
        ]) | llm | StrOutputParser()
    )


def extract_actions_items(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst. From the meeting transcript, 
        extract all the action items. For each provide:
         - Task description \n
         - Owner (who is responsible)\n
         - Deadline (if mentioned, else write 'not psecified')\n\n
         Format as as numbered list. If none found say 'No action items found"""
    )
    return chain.invoke(transcript)

def extract_key_decisions(transcript: str) -> str:
    chain = build_chain(
        """You are an expert meeting analyst. From the meeting transcript, 
        extract all key decisions made. Format as a numbered list. 
        If none found say 'No key decisions found.'"""
    )
    return chain.invoke(transcript)


def extract_questions(transcript: str) -> str:
    chain = build_chain(
        """From the meeting transcript, extract all unresolved questions
        or topics needing follow-up. Format as a numbered list.
        If none found say 'No open questions found.'"""
    )
    return chain.invoke(transcript)