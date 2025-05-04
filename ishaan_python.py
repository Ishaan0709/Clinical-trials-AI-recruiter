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

#########################
##### Helper Functions ##
#########################

def process_pdf(pdf_file):
    temp_path = "temp_trial.pdf"
    with open(temp_path, "wb") as f:
        f.write(pdf_file.read())

    loader = PyMuPDFLoader(temp_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore, documents

def extract_criteria(documents):
    llm = ChatOpenAI(temperature=0, openai_api_key=api_key)
    content = " ".join([doc.page_content for doc in documents])
    prompt = f"""
    From the following clinical trial criteria text, extract:
    - Minimum Age
    - Maximum Age
    - Required Condition
    - Required Biomarker
    - Allowed Stages
    - Allowed Gender
    - Exclude Diabetes (Yes/No)
    - Exclude Pregnant (Yes/No)

    Text:
    {content}

    Provide answer as JSON with keys:
    min_age, max_age, condition, biomarker, stages, gender, exclude_diabetes, exclude_pregnant.
    """
    response = llm.predict(prompt)
    try:
        criteria = json.loads(response)
    except:
        criteria = {}
    return criteria

def filter_df(df, criteria):
    filtered = df.copy()
    if "min_age" in criteria and criteria["min_age"]:
        filtered = filtered[filtered["Age"] >= int(criteria["min_age"])]
    if "max_age" in criteria and criteria["max_age"]:
        filtered = filtered[filtered["Age"] <= int(criteria["max_age"])]
    if "condition" in criteria and criteria["condition"]:
        filtered = filtered[filtered["Condition"].str.lower().str.contains(criteria["condition"].lower(), na=False)]
    if "biomarker" in criteria and criteria["biomarker"]:
        filtered = filtered[filtered["BiomarkerStatus"].str.contains(criteria["biomarker"], na=False)]
    if "stages" in criteria and criteria["stages"]:
        filtered = filtered[filtered["DiseaseStage"].isin(criteria["stages"] if isinstance(criteria["stages"], list) else [criteria["stages"]])]
    if "gender" in criteria and criteria["gender"] and criteria["gender"].lower() != "any":
        filtered = filtered[filtered["Gender"].str.lower() == criteria["gender"].lower()]
    if "exclude_diabetes" in criteria and str(criteria["exclude_diabetes"]).lower() == "yes":
        filtered = filtered[filtered["Diabetes"].str.lower() != "yes"]
    if "exclude_pregnant" in criteria and str(criteria["exclude_pregnant"]).lower() == "yes":
        filtered = filtered[filtered["Pregnant"].str.lower() != "yes"]
    return filtered

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
        st.session_state["admin_df"] = df
    elif "admin_df" in st.session_state:
        df = st.session_state["admin_df"]
    else:
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
            vectorstore, documents = process_pdf(pdf_file)
            st.success("PDF processed successfully.")
            criteria = extract_criteria(documents)
            st.session_state["criteria"] = criteria
            st.write("**Extracted Criteria:**")
            st.json(criteria)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            criteria = {}
            st.session_state["criteria"] = {}
    else:
        criteria = st.session_state.get("criteria", {})

    if not df.empty and criteria:
        filtered_m = filter_df(df, criteria)

        age_range_m = st.slider("Age Range", int(filtered_m["Age"].min()) if not filtered_m.empty else 0,
                                int(filtered_m["Age"].max()) if not filtered_m.empty else 100,
                                (int(df["Age"].min()), int(df["Age"].max())),
                                key="med_age")
        gender_m = st.selectbox("Gender", ["All"] + list(df["Gender"].dropna().unique()), key="med_gender")
        region_m = st.selectbox("Region", ["All"] + list(df["Region"].dropna().unique()), key="med_region")

        # Further filters
        final_filtered = filtered_m[(filtered_m["Age"] >= age_range_m[0]) & (filtered_m["Age"] <= age_range_m[1])]
        if gender_m != "All":
            final_filtered = final_filtered[final_filtered["Gender"] == gender_m]
        if region_m != "All":
            final_filtered = final_filtered[final_filtered["Region"] == region_m]

        st.markdown("### Eligible Volunteers (Privacy Protected)")

        display_cols = ["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender"]
        st.dataframe(final_filtered[display_cols])

        st.markdown("### Ask a Question (AI Powered)")
        med_question = st.text_input("Ask about eligible volunteers:", key="med_q")

        if "med_history" not in st.session_state:
            st.session_state["med_history"] = []

        if st.button("Ask", key="med_ask"):
            if med_question:
                data_context = final_filtered[display_cols].to_json(orient="records")
                full_context = f"PDF criteria:\n{criteria}\n\nData:\n{data_context}"
                answer = ask_llm(full_context, med_question)
                st.session_state["med_history"].append((med_question, answer))
                st.success(answer)

        with st.expander("Conversation History"):
            for q, a in st.session_state["med_history"]:
                st.markdown(f"**Q:** {q}")
                st.markdown(f"**A:** {a}")
                st.markdown("---")
    elif not criteria:
        st.warning("Please upload a Trial Criteria PDF to extract and apply eligibility rules.")

