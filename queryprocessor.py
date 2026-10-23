
"""
Query Processor - Online Pipeline
----------------------------------
Retrieves relevant document chunks from Pinecone
and generates an answer using Groq.
"""

from vectorstore import search_in_pinecone
from llm import query_llm_with_context


def process_user_query(query: str):
    """Retrieve document context and generate an answer."""

    # Step 1: Search Pinecone using the original question.
    matched_chunks = search_in_pinecone(query)

    # Step 2: Stop if no relevant document chunks are found.
    if not matched_chunks:
        print("No relevant document chunks found in Pinecone.")
        return

    # Step 3: Combine retrieved chunks into context text.
    context = "\n\n".join(matched_chunks)

    # Step 4: Generate the answer using the LLM.
    generated_response = query_llm_with_context(
        query,
        context,
        history=[]
    )

    print("\nAnswer:")
    print(generated_response)


if __name__ == "__main__":
    user_query = "What are the technical skills mentioned in the PDF?"
    process_user_query(user_query)
