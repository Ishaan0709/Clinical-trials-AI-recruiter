import streamlit as st
import pandas as pd
import fitz
import re
import os
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="AI Clinical Trial System", layout="wide", page_icon="⚕️")

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ---------------------- Core AI Components ----------------------
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)

# ---------------------- Session State Initialization ----------------------
if 'volunteers' not in st.session_state:
    st.session_state.update({
        'volunteers': pd.DataFrame(),
        'vector_store': None,
        'qa_chain': None,
        'dataset_qa': [],
        'trial_qa': [],
        'matched_df': None,
        'criteria': {}
    })

# ---------------------- Fixed Processing Functions ----------------------
def extract_criteria(text):
    """Extract inclusion/exclusion criteria from text"""
    criteria = {
        'min_age': 18,
        'max_age': 75,
        'biomarker': None,
        'stages': [],
        'exclude_diabetes': False,
        'exclude_pregnant': False
    }
    
    # Age extraction
    age_match = re.search(r'age\s*between\s*(\d+)\s*to\s*(\d+)', text, re.I)
    if age_match:
        criteria['min_age'] = int(age_match.group(1))
        criteria['max_age'] = int(age_match.group(2))
    
    # Biomarker extraction
    biomarker_match = re.search(r'biomarker\s*status:\s*(\w+\+?)', text, re.I)
    if biomarker_match:
        criteria['biomarker'] = biomarker_match.group(1).upper()
    
    # Stage extraction
    criteria['stages'] = re.findall(r'stage\s*(III?I?V?)', text, re.I)
    
    # Other criteria
    criteria['exclude_diabetes'] = bool(re.search(r'exclude\s*diabetes', text, re.I))
    criteria['exclude_pregnant'] = bool(re.search(r'exclude\s*pregnant', text, re.I))
    
    return criteria

def process_pdf(pdf_bytes):
    """Process PDF using LangChain pipeline with error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name
        
        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        
        st.session_state.vector_store = FAISS.from_documents(chunks, embeddings)
        st.session_state.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=st.session_state.vector_store.as_retriever()
        )
        
        # Extract criteria from text
        full_text = "\n".join([doc.page_content for doc in documents])
        st.session_state.criteria = extract_criteria(full_text)
        
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------- Core Application Logic ----------------------
def filter_volunteers(df):
    """Filter volunteers using AI-enhanced criteria"""
    try:
        filtered = df.copy()
        criteria = st.session_state.criteria
        
        # Basic filters
        filtered = filtered[
            (filtered['Age'] >= criteria['min_age']) &
            (filtered['Age'] <= criteria['max_age'])
        ]
        
        # Biomarker filter
        if criteria['biomarker']:
            filtered = filtered[filtered['BiomarkerStatus'] == criteria['biomarker']]
        
        # Stage filter
        if criteria['stages']:
            filtered = filtered[filtered['DiseaseStage'].isin(criteria['stages'])]
        
        # Medical exclusions
        if criteria['exclude_diabetes']:
            filtered = filtered[filtered['Diabetes'] == 'No']
        if criteria['exclude_pregnant']:
            filtered = filtered[filtered['Pregnant'] == 'No']
            
        return filtered
    
    except Exception as e:
        st.error(f"Filtering Error: {str(e)}")
        return pd.DataFrame()

# ---------------------- UI Components ----------------------
def main():
    st.title("AI-Powered Clinical Trial Matching System")
    
    # Data Loading
    if st.session_state.volunteers.empty:
        if os.path.exists("Realistic_Indian_Volunteers.csv"):
            try:
                st.session_state.volunteers = pd.read_csv("Realistic_Indian_Volunteers.csv")
                st.success("Loaded default volunteer data!")
            except Exception as e:
                st.error(f"CSV Load Error: {str(e)}")
        else:
            st.warning("Default dataset not found!")

    # Tab System
    tab_admin, tab_trial = st.tabs(["Admin Portal", "Trial Management"])

    with tab_admin:
        st.header("Volunteer Database Management")
        
        # CSV Upload
        csv_file = st.file_uploader("Upload Volunteer CSV", type=['csv'])
        if csv_file:
            try:
                st.session_state.volunteers = pd.read_csv(csv_file)
                st.success("Volunteer data updated!")
            except Exception as e:
                st.error(f"CSV Upload Error: {str(e)}")
        
        # Display Data
        if not st.session_state.volunteers.empty:
            st.dataframe(st.session_state.volunteers, height=400)

    with tab_trial:
        st.header("Trial Configuration")
        
        # PDF Upload
        pdf_file = st.file_uploader("Upload Trial PDF", type=['pdf'])
        if pdf_file:
            try:
                process_pdf(pdf_file.read())
                st.success("AI Processed Trial Document!")
                
                # Show extracted criteria
                with st.expander("Extracted Trial Criteria"):
                    st.json(st.session_state.criteria)
                
            except Exception as e:
                st.error(f"PDF Upload Error: {str(e)}")
        
        # Volunteer Matching
        if st.button("Find Eligible Volunteers"):
            if not st.session_state.volunteers.empty:
                st.session_state.matched_df = filter_volunteers(st.session_state.volunteers)
                if not st.session_state.matched_df.empty:
                    st.success(f"Found {len(st.session_state.matched_df)} eligible volunteers")
                    st.dataframe(st.session_state.matched_df)
                else:
                    st.warning("No eligible volunteers found")
            else:
                st.error("No volunteer data loaded!")

if __name__ == "__main__":
    main()
