import streamlit as st
import pandas as pd
import fitz
import re
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# ---------------------- Configuration ----------------------
st.set_page_config(page_title="AI Clinical Trial System", layout="wide", page_icon="⚕️")

# Load environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # Set in your system environment

# ---------------------- Core AI Components ----------------------
embeddings = OpenAIEmbeddings()
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# ---------------------- Session State Initialization ----------------------
if 'volunteers' not in st.session_state:
    st.session_state.update({
        'volunteers': pd.DataFrame(),
        'vector_store': None,
        'qa_chain': None,
        'criteria_text': "",
        'dataset_qa': [],
        'trial_qa': [],
        'matched_df': None
    })

# ---------------------- AI Processing Functions ----------------------
def process_pdf(pdf_bytes):
    """Process PDF using LangChain pipeline"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_bytes)
        loader = PyMuPDFLoader(tmp.name)
    
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
    os.unlink(tmp.name)

def ai_answer(question, context_df=None):
    """Hybrid AI + Rules Answering System"""
    # First check rule-based answers
    rule_based_answer = answer_question_about_df(question, context_df or st.session_state.volunteers)
    if "I'm sorry" not in rule_based_answer:
        return rule_based_answer
    
    # Fallback to AI if rules fail
    if st.session_state.qa_chain:
        result = st.session_state.qa_chain({"query": question})
        return result["result"]
    return "Unable to answer with current data"

# ---------------------- Original Core Functions (Enhanced) ----------------------
def parse_trial_pdf(pdf_bytes):
    """Integrated PDF Processing"""
    process_pdf(pdf_bytes)  # Using LangChain now
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return extract_criteria(text), [], []

def filter_volunteers(df, criteria):
    """Hybrid Filtering System"""
    # Original rule-based filtering
    filtered = df.copy()
    if criteria.get('min_age'):
        filtered = filtered[filtered['Age'] >= criteria['min_age']]
    if criteria.get('max_age'):
        filtered = filtered[filtered['Age'] <= criteria['max_age']]
    
    # AI-enhanced filtering
    if st.session_state.vector_store:
        docs = st.session_state.vector_store.similarity_search(criteria.get('biomarker', ""), k=3)
        biomarkers = [d.page_content for d in docs]
        filtered = filtered[filtered['BiomarkerStatus'].isin(biomarkers)]
    
    return filtered

# ---------------------- UI Components ----------------------
def main():
    st.title("AI-Powered Clinical Trial Matching System")
    
    # Data Loading
    if st.session_state.volunteers.empty:
        if os.path.exists("Realistic_Indian_Volunteers.csv"):
            try:
                st.session_state.volunteers = pd.read_csv("Realistic_Indian_Volunteers.csv")
            except Exception as e:
                st.error(f"Error loading CSV: {str(e)}")

    # Tab System
    tab_admin, tab_trial = st.tabs(["Admin Portal", "Trial Management"])

    with tab_admin:
        st.header("Volunteer Database Management")
        
        # File Upload
        csv_file = st.file_uploader("Upload Volunteer CSV", type=['csv'])
        if csv_file:
            st.session_state.volunteers = pd.read_csv(csv_file)
        
        # Filters and Table
        if not st.session_state.volunteers.empty:
            col1, col2 = st.columns(2)
            with col1:
                age_range = st.slider("Age Filter", 
                                    st.session_state.volunteers['Age'].min(),
                                    st.session_state.volunteers['Age'].max(),
                                    (25, 75))
            with col2:
                region_filter = st.selectbox("Region Filter", 
                                           ["All"] + list(st.session_state.volunteers['Region'].unique()))
            
            filtered_df = st.session_state.volunteers.query(f"Age >= {age_range[0]} & Age <= {age_range[1]}")
            if region_filter != "All":
                filtered_df = filtered_df[filtered_df['Region'] == region_filter]
            
            st.dataframe(filtered_df, height=400)
            
            # Q&A System
            with st.form("dataset_qa"):
                question = st.text_input("Ask about volunteers:")
                if st.form_submit_button("Get AI Analysis"):
                    answer = ai_answer(question, filtered_df)
                    st.session_state.dataset_qa.append((question, answer))

    with tab_trial:
        st.header("AI Trial Configuration")
        
        # PDF Processing
        pdf_file = st.file_uploader("Upload Trial PDF", type=['pdf'])
        if pdf_file:
            process_pdf(pdf_file.read())
            st.success("AI Processed Trial Document!")
        
        # Matching System
        if st.session_state.vector_store:
            question = st.text_input("Ask about trial criteria:")
            if question:
                answer = ai_answer(question)
                st.session_state.trial_qa.append((question, answer))
            
            if st.button("Find Eligible Volunteers"):
                docs = st.session_state.vector_store.similarity_search("eligibility criteria", k=5)
                criteria = "\n".join([d.page_content for d in docs])
                st.session_state.matched_df = filter_volunteers(st.session_state.volunteers, {
                    'min_age': 40,  # Example, extract from criteria
                    'max_age': 70,
                    'biomarker': criteria
                })
        
        if st.session_state.matched_df is not None:
            st.dataframe(st.session_state.matched_df)

    # History Section
    with st.expander("Conversation History"):
        st.subheader("Dataset Q&A")
        for q, a in st.session_state.dataset_qa[-5:]:
            st.markdown(f"**Q:** {q}  \n**A:** {a}")
        
        st.subheader("Trial Q&A")
        for q, a in st.session_state.trial_qa[-5:]:
            st.markdown(f"**Q:** {q}  \n**A:** {a}")

if __name__ == "__main__":
    main()
