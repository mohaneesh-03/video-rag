import os
import sys
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

from core.vector_store import search

load_dotenv()


def get_llm():
    return ChatGroq(
        model="openai/gpt-oss-120b",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2,
    )


def get_rephrase_chain(llm: ChatGroq):
    """
    Builds an LCEL chain to reformulate follow-up questions
    into standalone search queries based on chat history.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given a chat history and the latest user question which might reference "
                "context in the chat history, formulate a concise, standalone question "
                "that can be used to search a video transcript in a vector database. "
                "Do NOT answer the question. Only return the reformulated query. "
                "If the question is already standalone, return it as-is.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


def get_answer_chain(llm: ChatGroq):
    """
    Builds an LCEL chain to answer user questions grounded strictly
    in retrieved video transcript chunks.
    """
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert AI video assistant. Answer the user's question "
                "based ONLY on the video transcript context provided below.\n\n"
                "If the information is not present in the context, say: "
                "\"I could not find this information in the video transcript.\"\n\n"
                "Be concise, accurate, and helpful.\n\n"
                "Context from transcript:\n{context}",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    return prompt | llm | StrOutputParser()


class ConversationalRAG:
    """
    Stateful Conversational RAG Engine for Video Transcripts.
    """

    def __init__(self, top_k: int = 3, collection_name: str = "video_rag"):
        self.top_k = top_k
        self.collection_name = collection_name
        self.llm = get_llm()
        self.rephrase_chain = get_rephrase_chain(self.llm)
        self.answer_chain = get_answer_chain(self.llm)
        self.chat_history: List[BaseMessage] = []

    def ask(self, question: str, external_history: Optional[List[BaseMessage]] = None) -> Dict[str, Any]:
        """
        Processes a question through the conversational RAG pipeline.
        Can use internal self.chat_history or external history (e.g., Streamlit session state).
        """
        history = external_history if external_history is not None else self.chat_history

        # Step 1: Rephrase question if conversational history exists
        if history:
            standalone_query = self.rephrase_chain.invoke(
                {"chat_history": history, "question": question}
            ).strip()
        else:
            standalone_query = question

        # Step 2: Vector retrieval from Qdrant (with sentence-window context expansion)
        results = search(standalone_query, top_k=self.top_k, collection_name=self.collection_name)
        chunks = [
            result.payload.get("context", result.payload.get("text", ""))
            for result in results
            if result.payload and ("context" in result.payload or "text" in result.payload)
        ]
        context = "\n\n---\n\n".join(chunks)

        # Step 3: Grounded Answer Generation
        answer = self.answer_chain.invoke(
            {
                "context": context,
                "chat_history": history,
                "question": question,
            }
        ).strip()

        # Step 4: Record history (if using internal state)
        if external_history is None:
            self.chat_history.append(HumanMessage(content=question))
            self.chat_history.append(AIMessage(content=answer))

        return {
            "answer": answer,
            "query_used": standalone_query,
            "sources": chunks,
        }

    def clear_history(self):
        """Clears the conversational memory."""
        self.chat_history.clear()

    def cli_chat(self):
        """Interactive terminal chat loop."""
        print("\n" + "=" * 60)
        print("🎬 Video RAG Chatbot is online!")
        print("Ask any question about the video transcript.")
        print("Type 'exit', 'quit', or 'q' to end the session.")
        print("Type 'clear' to reset chat memory.")
        print("=" * 60)

        while True:
            try:
                user_input = input("\nYou: ").strip()
                if not user_input:
                    continue

                if user_input.lower() in ["exit", "quit", "q"]:
                    print("\n👋 Goodbye! Ending chat session.")
                    break

                if user_input.lower() == "clear":
                    self.clear_history()
                    print("\n🧹 Chat memory cleared.")
                    continue

                result = self.ask(user_input)
                print(f"\nAI: {result['answer']}")

            except KeyboardInterrupt:
                print("\n\nSession terminated by user.")
                break


# Backward-compatible helper for legacy single-turn callers
def ask_question(question: str) -> str:
    bot = ConversationalRAG(top_k=3)
    response = bot.ask(question)
    return response["answer"]


if __name__ == "__main__":
    bot = ConversationalRAG()
    bot.cli_chat()