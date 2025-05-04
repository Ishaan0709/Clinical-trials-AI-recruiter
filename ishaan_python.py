import streamlit as st
import pandas as pd
import fitz
import re
import os
import tempfile
from datetime import datetime
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
        'criteria': {},
        'trials': [],
        'current_trial': None
    })

# ---------------------- Processing Functions ----------------------
def extract_criteria(text):
    """Enhanced criteria extraction with medicine trial support"""
    criteria = {
        'min_age': 18, 'max_age': 75,
        'biomarker': None, 'stages': [],
        'exclude_diabetes': False, 'exclude_pregnant': False,
        'medicines': [], 'dosage_ranges': {}
    }
    
    # Medicine trial parameters
    medicine_matches = re.findall(r'medicine:\s*(\w+)\s*\(([\d-]+mg)\)', text, re.I)
    for med, dosage in medicine_matches:
        criteria['medicines'].append(med)
        criteria['dosage_ranges'][med] = dosage
    
    # Existing extraction logic
    age_match = re.search(r'age\s*between\s*(\d+)\s*to\s*(\d+)', text, re.I)
    if age_match:
        criteria['min_age'] = int(age_match.group(1))
        criteria['max_age'] = int(age_match.group(2))
    
    criteria['biomarker'] = re.search(r'biomarker\s*status:\s*(\w+\+?)', text, re.I)
    criteria['stages'] = re.findall(r'stage\s*(III?I?V?)', text, re.I)
    criteria['exclude_diabetes'] = bool(re.search(r'exclude\s*diabetes', text, re.I))
    criteria['exclude_pregnant'] = bool(re.search(r'exclude\s*pregnant', text, re.I))
    
    return criteria

def process_pdf(pdf_bytes):
    """Enhanced PDF processing with medicine trial tracking"""
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
        
        full_text = "\n".join([doc.page_content for doc in documents])
        st.session_state.criteria = extract_criteria(full_text)
        
        # Initialize new trial
        st.session_state.current_trial = {
            'medicines': st.session_state.criteria['medicines'],
            'start_date': datetime.now(),
            'participants': [],
            'results': {}
        }
        st.session_state.trials.append(st.session_state.current_trial)
        
    except Exception as e:
        st.error(f"PDF Processing Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# ---------------------- UI Components ----------------------
def show_qa_panel(context_df, history_key):
    """Universal Q/A Panel Component"""
    with st.expander("Ask Questions", expanded=True):
        question = st.text_input("Enter your question:", key=f"{history_key}_question")
        if st.button("Submit", key=f"{history_key}_submit"):
            if question:
                answer = ai_answer(question, context_df)
                st.session_state[history_key].append((question, answer))
        
        st.subheader("Conversation History")
        with st.container(height=300):
            for q, a in st.session_state[history_key][-5:]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.divider()

def volunteer_filters():
    """Fixed Volunteer Filtering Component"""
    with st.expander("Advanced Filters"):
        cols = st.columns(3)
        
        with cols[0]:
            gender_filter = st.multiselect(
                "Gender",
                options=["All"] + list(st.session_state.volunteers['Gender'].unique())
            )
        
        with cols[1]:
            stage_filter = st.multiselect(
                "Disease Stage",
                options=["All"] + list(st.session_state.volunteers['DiseaseStage'].unique())
            )
        
        with cols[2]:
            biomarker_filter = st.multiselect(
                "Biomarker Status",
                options=["All"] + list(st.session_state.volunteers['BiomarkerStatus'].unique())
            )
        
        return {
            'gender': gender_filter,
            'stage': stage_filter,
            'biomarker': biomarker_filter
        }

# ---------------------- Main Application ----------------------
def main():
    st.title("AI Clinical Trial Management System")
    
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
    tab_admin, tab_trial, tab_medicine = st.tabs([
        "Admin Portal", 
        "Trial Management", 
        "Live Medicine Trials"
    ])

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
        
        # Filters
        filters = volunteer_filters()
        filtered_df = st.session_state.volunteers.copy()
        
        if 'All' not in filters['gender']:
            filtered_df = filtered_df[filtered_df['Gender'].isin(filters['gender'])]
        if 'All' not in filters['stage']:
            filtered_df = filtered_df[filtered_df['DiseaseStage'].isin(filters['stage'])]
        if 'All' not in filters['biomarker']:
            filtered_df = filtered_df[filtered_df['BiomarkerStatus'].isin(filters['biomarker'])]
        
        # Data Display
        st.dataframe(filtered_df, height=400)
        
        # Q/A Panel
        show_qa_panel(filtered_df, 'dataset_qa')

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
                    
                    # Add to current trial
                    st.session_state.current_trial['participants'] = (
                        st.session_state.matched_df['VolunteerID'].tolist()
                    )
                else:
                    st.warning("No eligible volunteers found")
            else:
                st.error("No volunteer data loaded!")
        
        # Trial Q/A Panel
        show_qa_panel(st.session_state.matched_df, 'trial_qa')

if __name__ == "__main__":
    main()
