import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import os
import tempfile
import shutil
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Enhanced PDF validation
def is_valid_pdf(file_path):
    try:
        with fitz.open(file_path) as doc:
            if doc.is_encrypted:
                doc.authenticate("")  # Try empty password
            if doc.page_count == 0:
                return False
        return True
    except:
        return False

# Robust PDF processing with validation
def process_pdf(pdf_file):
    try:
        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "uploaded.pdf")
        
        with open(temp_path, "wb") as f:
            f.write(pdf_file.read())
        
        # Validate PDF
        if not is_valid_pdf(temp_path):
            raise ValueError("Invalid or corrupted PDF file")
        
        # Process with PyMuPDF
        loader = PyMuPDFLoader(temp_path)
        documents = loader.load()
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        return FAISS.from_documents(
            RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(documents),
            OpenAIEmbeddings(openai_api_key=openai_api_key)
        )
    
    except Exception as e:
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        st.error(f"PDF Processing Failed: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(page_title="Secure Clinical Trial System", layout="wide")

# TechVitals Admin Portal
with st.expander("TechVitals Admin Portal", expanded=True):
    uploaded_csv = st.file_uploader("Upload Volunteer CSV", type=["csv"])
    if uploaded_csv:
        try:
            df = pd.read_csv(uploaded_csv)
            required_cols = {'VolunteerID', 'Age', 'Gender', 'Condition', 'Email'}
            
            if required_cols.issubset(df.columns):
                st.success("Dataset loaded successfully!")
                st.dataframe(df.head())
            else:
                st.error(f"Missing columns: {required_cols - set(df.columns)}")
        except Exception as e:
            st.error(f"CSV Error: {str(e)}")

# Medical Company Portal
with st.expander("Medical Trial Portal", expanded=True):
    uploaded_pdf = st.file_uploader("Upload Trial Criteria PDF", 
                                  type=["pdf"],
                                  help="Upload a valid, unencrypted PDF document")
    
    if uploaded_pdf:
        vector_store = process_pdf(uploaded_pdf)
        
        if vector_store:
            st.success("PDF Successfully Analyzed!")
            
            # Eligibility Search
            query = st.text_input("Search for eligible volunteers:")
            if query:
                try:
                    results = vector_store.similarity_search(query, k=5)
                    st.subheader("Matching Criteria:")
                    for doc in results:
                        st.write(doc.page_content)
                except Exception as e:
                    st.error(f"Search Error: {str(e)}")

# Common error prevention features
st.markdown("""
<style>
    .stExpander {border: 1px solid #e0e0e0; border-radius: 8px; margin: 1rem 0;}
    .stFileUploader {margin-bottom: 1rem;}
</style>
""", unsafe_allow_html=True)
