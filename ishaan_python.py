import streamlit as st
import pandas as pd
import os
import json
import re
from dotenv import load_dotenv
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY not found in .env file.")
    st.stop()

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")

st.title("AI Clinical Trial Management System")
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

############################
##### Helper Functions #####
############################

def process_pdf(pdf_file):
    temp_path = "temp_trial.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyMuPDFLoader(temp_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore

def ask_llm(context, question):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    prompt = f"""
    You are a clinical trial assistant. Based on the following context and question, provide a clear answer.
    Context:
    {context}

    Question: {question}

    Give only the answer, no explanations.
    """
    response = llm.predict(prompt)
    return response.strip()

#########################
##### TechVitals Admin ##
#########################

with tab1:
    st.subheader("TechVitals Admin Portal")

    csv_file = st.file_uploader("Upload Volunteer CSV", type=["csv"], key="admin_csv")
    if csv_file:
        df = pd.read_csv(csv_file)
    else:
        if os.path.exists("patient_data1.csv"):
            df = pd.read_csv("patient_data1.csv")
            st.info("Using default patient_data1.csv")
        else:
            st.warning("No CSV uploaded and no default CSV found.")
            df = pd.DataFrame()

    if not df.empty:
        age_range = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())))
        gender = st.selectbox("Gender", ["All"] + list(df["Gender"].dropna().unique()))
        region = st.selectbox("Region", ["All"] + list(df["Region"].dropna().unique()))

        filtered = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
        if gender != "All":
            filtered = filtered[filtered["Gender"] == gender]
        if region != "All":
            filtered = filtered[filtered["Region"] == region]

        st.markdown("### Volunteer List")
        st.dataframe(filtered)

        st.markdown("### Ask a Question (AI Powered)")
        admin_question = st.text_input("Ask about volunteers:", key="admin_q")

        if "admin_history" not in st.session_state:
            st.session_state["admin_history"] = []

        if st.button("Ask", key="admin_ask"):
            if admin_question:
                context = filtered.to_json(orient="records")
                answer = ask_llm(context, admin_question)
                st.session_state["admin_history"].append((admin_question, answer))
                st.success(answer)

        with st.expander("Conversation History"):
            for q, a in st.session_state["admin_history"]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")

#########################
##### Medical Company ###
#########################

with tab2:
    st.subheader("Medical Company Portal")

    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"], key="med_pdf")
    if pdf_file:
        try:
            vectorstore = process_pdf(pdf_file)
            st.success("PDF processed and indexed.")
            st.session_state["vectorstore"] = vectorstore
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            st.session_state["vectorstore"] = None
    else:
        st.session_state["vectorstore"] = None

    if not df.empty:
        age_range_m = st.slider("Age Range", int(df["Age"].min()), int(df["Age"].max()), (int(df["Age"].min()), int(df["Age"].max())), key="med_age")
        gender_m = st.selectbox("Gender", ["All"] + list(df["Gender"].dropna().unique()), key="med_gender")
        region_m = st.selectbox("Region", ["All"] + list(df["Region"].dropna().unique()), key="med_region")

        filtered_m = df[(df["Age"] >= age_range_m[0]) & (df["Age"] <= age_range_m[1])]
        if gender_m != "All":
            filtered_m = filtered_m[filtered_m["Gender"] == gender_m]
        if region_m != "All":
            filtered_m = filtered_m[filtered_m["Region"] == region_m]

        st.markdown("### Eligible Volunteers (Privacy Protected)")

        # Only show ID, Email, Stage, Biomarker, Gender
        display_cols = ["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender"]
        st.dataframe(filtered_m[display_cols])

        st.markdown("### Ask a Question (AI Powered)")
        med_question = st.text_input("Ask about eligible volunteers:", key="med_q")

        if "med_history" not in st.session_state:
            st.session_state["med_history"] = []

        if st.button("Ask", key="med_ask"):
            if med_question:
                if st.session_state.get("vectorstore") is not None:
                    docs = st.session_state["vectorstore"].similarity_search(med_question, k=3)
                    context = " ".join([doc.page_content for doc in docs])
                else:
                    context = ""
                data_context = filtered_m[display_cols].to_json(orient="records")
                full_context = f"PDF context:\n{context}\n\nData:\n{data_context}"
                answer = ask_llm(full_context, med_question)
                st.session_state["med_history"].append((med_question, answer))
                st.success(answer)

        with st.expander("Conversation History"):
            for q, a in st.session_state["med_history"]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")
