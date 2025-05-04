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

# ---------------------- Data Loading & Validation ----------------------
def load_volunteer_data():
    """Safe data loading with validation"""
    if st.session_state.volunteers.empty:
        if os.path.exists("Realistic_Indian_Volunteers.csv"):
            try:
                df = pd.read_csv("Realistic_Indian_Volunteers.csv")
                
                # Validate required columns
                required_columns = [
                    'VolunteerID', 'Name', 'Age', 'Gender',
                    'Condition', 'DiseaseStage', 'BiomarkerStatus',
                    'Diabetes', 'Pregnant', 'Region'
                ]
                
                missing_cols = [col for col in required_columns if col not in df.columns]
                
                if missing_cols:
                    st.error(f"Missing columns: {', '.join(missing_cols)}")
                    return pd.DataFrame()
                
                st.session_state.volunteers = df
                return df
                
            except Exception as e:
                st.error(f"CSV Error: {str(e)}")
                return pd.DataFrame()
        else:
            st.warning("Default dataset not found!")
            return pd.DataFrame()
    return st.session_state.volunteers

# ---------------------- PDF Processing ----------------------
def process_pdf(pdf_bytes):
    """Process PDF with error handling"""
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
        extract_criteria(full_text)
        
        st.session_state.current_trial = {
            'start_date': datetime.now(),
            'participants': [],
            'results': {}
        }
        
    except Exception as e:
        st.error(f"PDF Error: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def extract_criteria(text):
    """Extract trial criteria from text"""
    criteria = {
        'min_age': 18, 'max_age': 75,
        'biomarker': None, 'stages': [],
        'exclude_diabetes': False, 
        'exclude_pregnant': False,
        'medicines': []
    }
    
    try:
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
        
        # Medical exclusions
        criteria['exclude_diabetes'] = bool(re.search(r'exclude\s*diabetes', text, re.I))
        criteria['exclude_pregnant'] = bool(re.search(r'exclude\s*pregnant', text, re.I))
        
        # Medicine trials
        medicine_matches = re.findall(r'medicine:\s*(\w+)\s*\(([\d-]+mg)\)', text, re.I)
        criteria['medicines'] = [med for med, _ in medicine_matches]
        
    except Exception as e:
        st.error(f"Criteria Extraction Error: {str(e)}")
    
    st.session_state.criteria = criteria

# ---------------------- Filtering System ----------------------
def volunteer_filters():
    """Safe filtering with column checks"""
    filters = {
        'gender': [],
        'stage': [],
        'biomarker': []
    }
    
    try:
        df = st.session_state.volunteers
        with st.expander("🔍 Advanced Filters"):
            cols = st.columns(3)
            
            # Gender Filter
            with cols[0]:
                if 'Gender' in df.columns:
                    options = ["All"] + list(df['Gender'].astype(str).unique())
                    filters['gender'] = st.multiselect(
                        "Gender", options, default=["All"]
                    )
            
            # Disease Stage Filter
            with cols[1]:
                if 'DiseaseStage' in df.columns:
                    options = ["All"] + list(df['DiseaseStage'].astype(str).unique())
                    filters['stage'] = st.multiselect(
                        "Disease Stage", options, default=["All"]
                    )
            
            # Biomarker Filter
            with cols[2]:
                if 'BiomarkerStatus' in df.columns:
                    options = ["All"] + list(df['BiomarkerStatus'].astype(str).unique())
                    filters['biomarker'] = st.multiselect(
                        "Biomarker Status", options, default=["All"]
                    )
    
    except Exception as e:
        st.error(f"Filter Error: {str(e)}")
    
    return filters

def apply_filters(filters):
    """Apply filters to volunteer data"""
    try:
        df = st.session_state.volunteers.copy()
        
        # Gender filter
        if 'All' not in filters['gender']:
            df = df[df['Gender'].isin(filters['gender'])]
        
        # Stage filter
        if 'All' not in filters['stage']:
            df = df[df['DiseaseStage'].isin(filters['stage'])]
        
        # Biomarker filter
        if 'All' not in filters['biomarker']:
            df = df[df['BiomarkerStatus'].isin(filters['biomarker'])]
        
        return df
    
    except Exception as e:
        st.error(f"Filtering Error: {str(e)}")
        return pd.DataFrame()

# ---------------------- Q/A System ----------------------
def ai_answer(question, context_df=None):
    """Hybrid Q/A System"""
    try:
        # Rule-based answers
        if context_df is not None and not context_df.empty:
            if "list" in question.lower() or "show" in question.lower():
                return context_df.to_markdown()
            
            if "count" in question.lower():
                return f"Total volunteers: {len(context_df)}"
        
        # AI answers
        if st.session_state.qa_chain:
            result = st.session_state.qa_chain({"query": question})
            return result["result"]
        
        return "Answer not available"
    
    except Exception as e:
        return f"Error answering question: {str(e)}"

# ---------------------- UI Components ----------------------
def show_qa_panel(context_df, history_key):
    """Interactive Q/A Panel"""
    with st.expander("💬 Ask Questions"):
        question = st.text_input("Your question:", key=f"{history_key}_input")
        
        if st.button("Submit", key=f"{history_key}_submit"):
            if question:
                answer = ai_answer(question, context_df)
                st.session_state[history_key].append((question, answer))
        
        st.subheader("History")
        for q, a in st.session_state[history_key][-5:]:
            st.markdown(f"**Q:** {q}  \n**A:** {a}")
            st.divider()

# ---------------------- Main Application ----------------------
def main():
    st.title("AI Clinical Trial Management System")
    
    # Load data
    df = load_volunteer_data()
    
    if df.empty:
        csv_file = st.file_uploader("Upload Volunteer CSV", type=['csv'])
        if csv_file:
            try:
                st.session_state.volunteers = pd.read_csv(csv_file)
                st.rerun()
            except Exception as e:
                st.error(f"Upload Error: {str(e)}")
        return

    # Tab System
    tab1, tab2, tab3 = st.tabs(["Admin Portal", "Trial Manager", "Medicine Trials"])

    with tab1:
        st.header("Volunteer Management")
        
        # Filters
        filters = volunteer_filters()
        filtered_df = apply_filters(filters)
        
        # Data Display
        st.dataframe(filtered_df, use_container_width=True, height=400)
        
        # Q/A Panel
        show_qa_panel(filtered_df, 'dataset_qa')

    with tab2:
        st.header("Trial Configuration")
        
        # PDF Upload
        pdf_file = st.file_uploader("Upload Trial PDF", type=['pdf'])
        if pdf_file:
            process_pdf(pdf_file.read())
            st.success("PDF Processed Successfully!")
        
        # Criteria Display
        if st.session_state.criteria:
            with st.expander("Trial Criteria"):
                st.json(st.session_state.criteria)
        
        # Volunteer Matching
        if st.button("Find Eligible Volunteers"):
            try:
                filtered = st.session_state.volunteers[
                    (st.session_state.volunteers['Age'] >= st.session_state.criteria['min_age']) &
                    (st.session_state.volunteers['Age'] <= st.session_state.criteria['max_age'])
                ]
                
                if st.session_state.criteria['biomarker']:
                    filtered = filtered[filtered['BiomarkerStatus'] == st.session_state.criteria['biomarker']]
                
                st.session_state.matched_df = filtered
                st.success(f"Found {len(filtered)} eligible volunteers")
                
            except Exception as e:
                st.error(f"Matching Error: {str(e)}")
        
        if st.session_state.matched_df is not None:
            st.dataframe(st.session_state.matched_df)
            show_qa_panel(st.session_state.matched_df, 'trial_qa')

    with tab3:
        st.header("Live Medicine Trials")
        if st.session_state.current_trial:
            st.subheader("Active Trial Details")
            cols = st.columns(3)
            cols[0].metric("Medicines", len(st.session_state.criteria['medicines']))
            cols[1].metric("Participants", len(st.session_state.current_trial['participants']))
            
            # Trial Management
            selected_med = st.selectbox("Select Medicine", options=st.session_state.criteria['medicines'])
            dosage = st.slider("Dosage (mg)", 0, 1000, (50, 200))
            
            if st.button("Update Trial"):
                st.session_state.current_trial['results'][selected_med] = {
                    'dosage': dosage,
                    'participants': st.session_state.matched_df['VolunteerID'].tolist()
                }
                st.success("Trial Updated!")
        else:
            st.warning("No active trial - Upload PDF first")

if __name__ == "__main__":
    main()
