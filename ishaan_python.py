import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
import json
import re

# Page configuration for wide layout and title
st.set_page_config(page_title="AI Clinical Trial Volunteer Recruiter", layout="wide")

# Load OpenAI API key from environment
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OpenAI API key not found. Please set OPENAI_API_KEY in the .env file.")
    st.stop()

st.title("AI Clinical Trial Volunteer Recruiter")
st.subheader("TechVitals Admin    |    Medical Company")

# 1. PDF Upload and Criteria Extraction
st.markdown("#### 1. Upload Clinical Trial PDF")
pdf_file = st.file_uploader("Upload a PDF with Trial Inclusion/Exclusion Criteria", type=["pdf"])
if pdf_file:
    try:
        pdf_bytes = pdf_file.read()
        # Save PDF to a temporary file for loader
        temp_pdf_path = "temp_trial.pdf"
        with open(temp_pdf_path, "wb") as f:
            f.write(pdf_bytes)
        loader = PyMuPDFLoader(temp_pdf_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        documents = loader.load_and_split(text_splitter=text_splitter)
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        vectorstore = FAISS.from_documents(documents, embeddings)
        st.session_state["vectorstore"] = vectorstore
        st.success("PDF processed and trial criteria indexed.")
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        st.session_state.pop("vectorstore", None)
else:
    st.session_state.pop("vectorstore", None,)

# 2. Load Volunteer Dataset (CSV) and apply basic filters
st.markdown("#### 2. Load Volunteer Dataset (CSV)")
vol_file = st.file_uploader("Upload Volunteer CSV (or use sample dataset)", type=["csv"])
if vol_file:
    try:
        df_vol = pd.read_csv(vol_file)
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        df_vol = pd.DataFrame()
else:
    sample_csv = "patient_data1.csv"
    if os.path.exists(sample_csv):
        df_vol = pd.read_csv(sample_csv)
        st.info("Using sample dataset from patient_data1.csv")
    else:
        df_vol = pd.DataFrame()

if not df_vol.empty:
    required_cols = {"Age", "Gender", "Region", "Condition"}
    if not required_cols.issubset(df_vol.columns):
        st.error(f"Dataset is missing required columns: {required_cols}")
    else:
        age_min_val = int(df_vol["Age"].min()) if "Age" in df_vol else 0
        age_max_val = int(df_vol["Age"].max()) if "Age" in df_vol else 100
        age_min, age_max = st.slider("Age Range", min_value=age_min_val, max_value=age_max_val, value=(age_min_val, age_max_val))
        gender_options = ["All"] + sorted([str(g) for g in df_vol["Gender"].dropna().unique()])
        gender_choice = st.selectbox("Gender", gender_options)
        region_options = ["All"] + sorted([str(r) for r in df_vol["Region"].dropna().unique()])
        region_choice = st.selectbox("Region", region_options)

        # Apply UI filters to dataset
        df_filtered_ui = df_vol.copy()
        df_filtered_ui = df_filtered_ui[(df_filtered_ui["Age"] >= age_min) & (df_filtered_ui["Age"] <= age_max)]
        if gender_choice != "All":
            df_filtered_ui = df_filtered_ui[df_filtered_ui["Gender"].astype(str).str.lower() == gender_choice.lower()]
        if region_choice != "All":
            df_filtered_ui = df_filtered_ui[df_filtered_ui["Region"].astype(str).str.lower() == region_choice.lower()]

        # 3. Question-Based Filtering using AI
        question = st.text_input("Ask a question about trial criteria (e.g., 'female patients with no heart issues above 50')")

        # Display current volunteer list after UI filters
        st.markdown("### Volunteer List (after applying filters above)")
        if df_filtered_ui.empty:
            st.write("No volunteers available with current filters.")
        else:
            st.dataframe(df_filtered_ui.reset_index(drop=True))

        # When user clicks search, perform semantic filtering based on question
        if st.button("Find Eligible Volunteers"):
            if "vectorstore" not in st.session_state:
                st.error("Please upload a trial PDF to extract criteria before searching.")
            elif question.strip() == "":
                st.error("Please enter a question to filter by trial criteria.")
            else:
                # Retrieve relevant criteria text from the PDF via semantic search
                try:
                    docs = st.session_state["vectorstore"].similarity_search(question, k=5)
                except Exception as e:
                    st.error(f"Error during semantic search: {e}")
                    docs = []
                context_text = " ".join([doc.page_content for doc in docs])

                # Use ChatOpenAI to interpret question and criteria into filters
                llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
                prompt = (
                    "You are an expert clinical trial assistant. Given the trial criteria and a user query, extract relevant patient filters.\n"
                    f"Trial Criteria:\n{context_text}\n\n"
                    f"User Query: {question}\n\n"
                    "Output a JSON object with keys among: min_age, max_age, gender, region, include_conditions, exclude_conditions."
                )
                criteria = {}
                try:
                    response = llm.predict(prompt)
                    criteria = json.loads(response.strip())
                except Exception:
                    # Fallback to simple keyword parsing if LLM fails
                    q = question.lower()
                    if "female" in q:
                        criteria["gender"] = "Female"
                    elif "male" in q:
                        criteria["gender"] = "Male"
                    age_nums = re.findall(r"\d+", q)
                    if age_nums:
                        age_val = int(age_nums[0])
                        if any(word in q for word in ["above", "over", "older than"]):
                            criteria["min_age"] = age_val
                        elif any(word in q for word in ["below", "under", "younger than"]):
                            criteria["max_age"] = age_val
                    if "no " in q or "without" in q:
                        terms = []
                        if "no " in q:
                            terms.append(q.split("no ", 1)[1].split()[0])
                        if "without" in q:
                            terms.append(q.split("without", 1)[1].split()[0])
                        if terms:
                            criteria["exclude_conditions"] = terms
                    if "with " in q and "no " not in q and "without" not in q:
                        term = q.split("with ", 1)[1].split()[0]
                        criteria.setdefault("include_conditions", []).append(term)

                # Apply both UI and question-based filters
                df_final = df_filtered_ui.copy()
                if "min_age" in criteria:
                    df_final = df_final[df_final["Age"] >= int(criteria["min_age"])]
                if "max_age" in criteria:
                    df_final = df_final[df_final["Age"] <= int(criteria["max_age"])]
                if "gender" in criteria:
                    df_final = df_final[df_final["Gender"].astype(str).str.lower() == str(criteria["gender"]).lower()]
                if "region" in criteria:
                    region_val = str(criteria["region"]).lower()
                    df_final = df_final[df_final["Region"].astype(str).str.lower().str.contains(region_val)]
                if "include_conditions" in criteria:
                    for cond in criteria["include_conditions"]:
                        df_final = df_final[df_final["Condition"].astype(str).str.lower().str.contains(str(cond).lower())]
                if "exclude_conditions" in criteria:
                    for cond in criteria["exclude_conditions"]:
                        df_final = df_final[~df_final["Condition"].astype(str).str.lower().str.contains(str(cond).lower())]

                # 4. Display eligible patients
                st.markdown("### Eligible Volunteers")
                if df_final.empty:
                    st.write("No volunteers match the trial criteria for this query.")
                else:
                    for _, row in df_final.iterrows():
                        name = row.get("Name", "N/A")
                        vol_id = row.get("VolunteerID", "")
                        title = f"{name} (ID: {vol_id})"
                        with st.expander(title):
                            st.write(f"**Age:** {row.get('Age', '')}")
                            st.write(f"**Gender:** {row.get('Gender', '')}")
                            st.write(f"**Region:** {row.get('Region', '')}")
                            st.write(f"**Condition:** {row.get('Condition', '')}")
                            email = row.get("Email", "")
                            if pd.notna(email) and email:
                                st.write(f"**Email:** {email}")
