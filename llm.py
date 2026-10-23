
"""
LLM Query Module
Uses Groq to answer questions based on retrieved document context.
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY is missing. Add it to your .env file."
    )

client = OpenAI(
    api_key=api_key,
    base_url="https://api.groq.com/openai/v1"
)

MODEL = "openai/gpt-oss-20b"


def query_llm_with_context(query: str, context: str, history):
    """Generate an answer using the retrieved document context."""

    system_context = """
You are an intelligent document Q&A assistant.

Answer using the supplied document context.
You may infer reasonably from relevant information in the context.
If there is absolutely no relevant information, respond exactly:
"I don't have enough information in the provided document to answer this."

Be concise, accurate, and natural.
"""

    messages = [
        {"role": "system", "content": system_context}
    ]

    # Preserve recent conversation history.
    messages.extend(history[-10:])

    messages.append({
        "role": "user",
        "content": f"""
Context:
{context}

Question:
{query}
"""
    })

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.6
    )

    return response.choices[0].message.content
