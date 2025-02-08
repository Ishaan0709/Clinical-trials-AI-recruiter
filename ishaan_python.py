import requests
import pandas as pd
import streamlit as st
import os
import tempfile
import time

from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Load environment variables from .env file
load_dotenv()

# Check API Key from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
    raise ValueError("Invalid or missing OpenAI API key. Please check your .env file.")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings()

def load_documents(uploaded_files):
    """Loads and processes both PDF and CSV files into a FAISS vector store."""
    rec_chunks = []
    
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(temp_file_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100, length_function=len, add_start_index=True
            )
            chunks = text_splitter.split_documents(documents)
            rec_chunks.extend(chunks)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(temp_file_path)
            text_data = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
            doc = Document(page_content=text_data)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=100, length_function=len, add_start_index=True
            )
            chunks = text_splitter.split_documents([doc])
            rec_chunks.extend(chunks)

    # Convert document chunks to embeddings
    embedding_function = OpenAIEmbeddings()
    faiss_index = FAISS.from_documents(rec_chunks, embedding_function)

    return faiss_index  # Return FAISS index instead of raw documents

def main():
    st.title("Clinical Trials Recruitment Insights Bot")

    files = st.file_uploader("Upload PDF & CSV Data", type=["pdf", "csv"], accept_multiple_files=True)

    if 'conversation' not in st.session_state:
        st.session_state.conversation = []

    if 'faiss_index' not in st.session_state:
        st.session_state.faiss_index = None

    if 'user_input' not in st.session_state:
        st.session_state.user_input = ""

    if files and st.button("Submit Documents"):
        try:
            st.session_state.faiss_index = load_documents(files)
            st.session_state.conversation.append(("System", "Documents uploaded and processed successfully."))
        except RuntimeError as e:
            st.session_state.conversation.append(("System", f"Error: {str(e)}"))

    if st.session_state.faiss_index:
        user_input = st.text_input("Ask a question about the documents:", value="", key="user_input_placeholder")

        if st.button("Ask"):
            if user_input:
                st.session_state.conversation.append(("You", user_input))

                model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

                # Retrieve relevant document chunks dynamically based on query type
                retrieved_docs = st.session_state.faiss_index.similarity_search(user_input, k=10)

                # Apply question-specific filtering
                filtered_docs = []
                if "Ibrutinib" in user_input or "Nivolumab" in user_input:
                    filtered_docs = [doc for doc in retrieved_docs if "Ibrutinib" in doc.page_content or "Nivolumab" in doc.page_content]
                elif "pediatric" in user_input.lower() or "children" in user_input.lower():
                    filtered_docs = [doc for doc in retrieved_docs if "pediatric" in doc.page_content.lower() or "children" in doc.page_content.lower()]
                elif "Phase 2" in user_input or "Phase 3" in user_input:
                    filtered_docs = [doc for doc in retrieved_docs if "Phase 2" in doc.page_content or "Phase 3" in doc.page_content]
                elif "inclusion" in user_input.lower() or "exclusion" in user_input.lower():
                    filtered_docs = [doc for doc in retrieved_docs if "inclusion" in doc.page_content.lower() or "exclusion" in doc.page_content.lower()]
                elif "primary endpoint" in user_input.lower():
                    filtered_docs = [doc for doc in retrieved_docs if "primary endpoint" in doc.page_content.lower()]
                else:
                    filtered_docs = retrieved_docs  # Use default if no special filtering needed

                if not filtered_docs:
                    assistant_response = "No relevant trials found matching your query."
                else:
                    retrieved_texts = [doc.page_content for doc in filtered_docs]
                    context = "\n".join(retrieved_texts)

                    # Improved Prompt for Structured Responses
                    response = model.invoke(
                        f"Provide structured answers using bullet points or tables for clarity. "
                        f"Ensure that inclusion and exclusion criteria are clearly separated. "
                        f"List all primary endpoints where applicable."
                        f"\n\nContext: {context}\n\nQuestion: {user_input}\n\nAnswer:"
                    )
                    assistant_response = response.content if hasattr(response, "content") else str(response)

                st.session_state.conversation.append(("AI", assistant_response))
                st.session_state.user_input = ""
                st.rerun()

    for speaker, message in st.session_state.conversation:
        st.write(f"{speaker}: {message}")

if __name__ == "__main__":
    main()