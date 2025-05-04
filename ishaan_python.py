import streamlit as st
import pandas as pd
import os
import tempfile
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import re
from dotenv import load_dotenv

# Configuration
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
st.set_page_config(page_title="AI Clinical Trial System", layout="wide")

# Initialize session state
if 'tech_history' not in st.session_state:
    st.session_state.tech_history = []
if 'medical_history' not in st.session_state:
    st.session_state.medical_history = []

# Helper functions
def process_pdf(pdf_file):
    """Process PDF and return vector store"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.read())
        loader = PyMuPDFLoader(tmp.name)
        documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    return FAISS.from_documents(docs, OpenAIEmbeddings(openai_api_key=openai_api_key))

def ai_query(question, context, history_key):
    """Handle AI queries with history"""
    chat = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    response = chat([
        SystemMessage(content=f"Context: {context}"),
        HumanMessage(content=question)
    ])
    st.session_state[history_key].append((question, response.content))
    return response.content

# Main App
st.title("AI Clinical Trial Management System")
tech_tab, medical_tab = st.tabs(["TechVitals Admin", "Medical Company"])

with tech_tab:
    st.header("TechVitals Admin Portal")
    
    # CSV Upload and Filters
    csv_file = st.file_uploader("Upload Volunteer CSV", type=["csv"])
    if csv_file:
        df = pd.read_csv(csv_file)
        required_cols = {"VolunteerID", "Age", "Gender", "Region", "Condition", "Email"}
        
        if required_cols.issubset(df.columns):
            # Filters
            col1, col2, col3 = st.columns(3)
            with col1:
                age_range = st.slider("Age Range", 
                                    int(df.Age.min()), 
                                    int(df.Age.max()), 
                                    (25, 60))
            with col2:
                gender = st.selectbox("Gender", ["All"] + list(df.Gender.unique()))
            with col3:
                region = st.selectbox("Region", ["All"] + list(df.Region.unique()))
            
            # Apply filters
            filtered_df = df[
                (df.Age >= age_range[0]) & 
                (df.Age <= age_range[1])
            ]
            if gender != "All":
                filtered_df = filtered_df[filtered_df.Gender == gender]
            if region != "All":
                filtered_df = filtered_df[filtered_df.Region == region]
            
            # Display filtered data
            st.dataframe(filtered_df[['VolunteerID', 'Age', 'Gender', 'Region', 'Condition']])
            
            # Q&A Section
            st.subheader("Dataset Q&A")
            question = st.text_input("Ask about volunteers:")
            if st.button("Ask"):
                context = filtered_df.to_csv()
                answer = ai_query(question, context, 'tech_history')
                st.write(answer)
            
            # Conversation History
            with st.expander("Conversation History"):
                for q, a in st.session_state.tech_history[-5:]:
                    st.markdown(f"**Q:** {q}  \n**A:** {a}")

with medical_tab:
    st.header("Medical Company Portal")
    
    # PDF Upload and Processing
    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
    if pdf_file:
        vectorstore = process_pdf(pdf_file)
        
        # Eligibility Check
        st.subheader("Find Eligible Volunteers")
        query = st.text_input("Enter eligibility criteria:")
        
        if st.button("Check Eligibility"):
            # AI Processing
            docs = vectorstore.similarity_search(query, k=5)
            context = "\n".join([d.page_content for d in docs])
            
            # Get filtered volunteers
            filtered = df.copy()  # Assuming df is loaded from TechVitals
            
            # AI-based filtering
            prompt = f"""Analyze trial criteria and filter volunteers:
            Criteria: {context}
            Volunteers Metadata: {filtered[['Age', 'Gender', 'Condition']].to_csv()}
            Return JSON with filters to apply"""
            
            response = ChatOpenAI().predict(prompt)
            try:
                filters = json.loads(response)
                if 'min_age' in filters: filtered = filtered[filtered.Age >= filters['min_age']]
                if 'max_age' in filters: filtered = filtered[filtered.Age <= filters['max_age']]
                if 'gender' in filters: filtered = filtered[filtered.Gender == filters['gender']]
                if 'conditions' in filters: 
                    filtered = filtered[filtered.Condition.isin(filters['conditions'])]
                
                # Display confidential results
                st.subheader("Eligible Volunteers")
                st.dataframe(filtered[['VolunteerID', 'Age', 'Gender', 'Condition', 'Email']])
            
            except Exception as e:
                st.error(f"Error applying filters: {e}")
        
        # Medical Q&A
        st.subheader("Trial Criteria Q&A")
        med_question = st.text_input("Ask about trial criteria:")
        if st.button("Ask Expert"):
            answer = ai_query(med_question, context, 'medical_history')
            st.write(answer)
        
        # Medical History
        with st.expander("Trial Conversation History"):
            for q, a in st.session_state.medical_history[-5:]:
                st.markdown(f"**Q:** {q}  \n**A:** {a}")
