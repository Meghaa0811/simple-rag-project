import streamlit as st
import requests

st.set_page_config(
    page_title="Simple RAG Chatbot",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Simple RAG Chatbot")

if st.sidebar.button("🗑️ Clear Chat"):

    response = requests.post(
        "http://127.0.0.1:8000/clear-chat"
    )

    if response.status_code == 200:

        st.session_state.messages = []

        st.success("Chat cleared successfully!")

        st.rerun()

# ==========================
# PDF Upload Section
# ==========================

uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    if st.button("Process PDF"):

        with st.spinner("Processing PDF..."):

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/pdf"
                )
            }

            response = requests.post(
                "http://127.0.0.1:8000/upload-pdf",
                files=files
            )

            if response.status_code == 200:
                st.success("PDF processed successfully!")
                st.json(response.json())
            else:
                st.error("Failed to process PDF")


st.divider()

# ==========================
# Chat Section
# ==========================

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_question = st.chat_input("Ask anything about the uploaded PDF...")

if user_question:

    # Show user message
    st.session_state.messages.append(
        {"role": "user", "content": user_question}
    )

    with st.chat_message("user"):
        st.markdown(user_question)

    # Send question to backend
    with st.spinner("Thinking..."):

        response = requests.post(
            "http://127.0.0.1:8000/ask",
            json={
                "question": user_question
            }
        )

        if response.status_code == 200:

            result = response.json()
            answer = result["answer"]

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

            with st.chat_message("assistant"):
                st.markdown(answer)
        else:
            st.error("Failed to get answer from backend")