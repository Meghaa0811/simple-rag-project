from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import os

from dataprocessor import process_pdf
from embedder import embed_user_query
from vectorstore import search_in_pinecone
from llm import query_llm_with_context

app = FastAPI()

chat_history = []


# ==========================
# Request Model
# ==========================

class QueryRequest(BaseModel):
    question: str


# ==========================
# PDF Upload Endpoint
# ==========================

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    global chat_history
    chat_history.clear()

    os.makedirs("data", exist_ok=True)

    file_path = f"data/{file.filename}"

    with open(file_path, "wb") as f:
        f.write(await file.read())

    result = process_pdf(file_path)

    return {
        "message": "PDF processed successfully",
        "details": result
    }


# ==========================
# Question Answer Endpoint
# ==========================

@app.post("/ask")
def ask_question(request: QueryRequest):

    query_embedding = embed_user_query(
        request.question
    )

    matched_chunks = search_in_pinecone(
        query_embedding
    )

    context = "\n\n".join(matched_chunks)

    print("\nCONTEXT SENT TO LLM:")
    print(context)

    # Save user question first
    chat_history.append(
        {
            "role": "user",
            "content": request.question
        }
    )

    # Generate answer
    answer = query_llm_with_context(
        request.question,
        context,
        chat_history
    )

    # Save assistant answer
    chat_history.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    print("\nCHAT HISTORY:")
    print(chat_history)

    return {
        "question": request.question,
        "answer": answer
    }

@app.post("/clear-chat")
def clear_chat():

    global chat_history

    chat_history.clear()

    return {
        "message": "Chat history cleared successfully"
    }

# ==========================
# Health Check
# ==========================

@app.get("/")
def home():
    return {
        "message": "Simple RAG API Running"
    }