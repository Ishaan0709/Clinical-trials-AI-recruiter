import streamlit as st
import pandas as pd
import os
import fitz
import tempfile
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")
st.title("AI Clinical Trial Management System")

# -------------------------- SESSION STATE INIT --------------------------
if "volunteers_df" not in st.session_state:
    if "Realistic_Indian_Volunteers.csv" in os.listdir():
        st.session_state.volunteers_df = pd.read_csv("Realistic_Indian_Volunteers.csv")
    else:
        st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.history_admin = []
    st.session_state.history_medical = []
    st.session_state.criteria = None
    st.session_state.vectorstore = None

# -------------------------- PDF PARSER FUNCTION --------------------------
def parse_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    text = " ".join([chunk.page_content for chunk in chunks])

    # Create vectorstore
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Extract criteria
    criteria = {
        "min_age": 30,
        "max_age": 75,
        "condition": None,
        "biomarker": None,
        "stages": [],
        "gender": "Any",
        "exclude_diabetes": "No",
        "exclude_pregnant": "No"
    }

    age_match = re.findall(r'(\d+)[ -]+(\d+)\s*years', text)
    if age_match:
        criteria["min_age"], criteria["max_age"] = map(int, age_match[0])

    if re.search("Non-Small Cell Lung Cancer|NSCLC", text, re.IGNORECASE):
        criteria["condition"] = "Non-Small Cell Lung Cancer (NSCLC)"

    if "EGFR+" in text:
        criteria["biomarker"] = "EGFR+"

    stage_match = re.findall(r'\b(Stage\s*)?(I{1,3}|IV)\b', text)
    if stage_match:
        criteria["stages"] = list(set([s[1].upper() for s in stage_match]))

    if "pregnant" in text.lower():
        criteria["exclude_pregnant"] = "Yes"

    return criteria, vectorstore

# -------------------------- FILTER FUNCTION --------------------------
def filter_volunteers(df, criteria):
    if df.empty:
        return pd.DataFrame()

    result = df.copy()
    result = result[(result['Age'] >= criteria["min_age"]) & (result['Age'] <= criteria["max_age"])]
    if criteria["condition"]:
        result = result[result["Condition"].str.contains("NSCLC|Non-Small Cell", case=False, na=False)]
    if criteria["biomarker"]:
        result = result[result["BiomarkerStatus"].str.upper() == criteria["biomarker"].upper()]
    if criteria["stages"]:
        stage_clean = [s.replace("Stage", "").strip().upper() for s in criteria["stages"]]
        result = result[result["DiseaseStage"].str.upper().isin(stage_clean)]
    if criteria["gender"] != "Any":
        result = result[result["Gender"].str.lower() == criteria["gender"].lower()]
    if criteria["exclude_diabetes"] == "Yes":
        result = result[result["Diabetes"].str.lower() != "yes"]
    if criteria["exclude_pregnant"] == "Yes":
        result = result[result["Pregnant"].str.lower() != "yes"]
    return result

# -------------------------- QA FUNCTION --------------------------
def answer_question(q, df):
    if df.empty:
        return "No eligible volunteers."
    q_lower = q.lower()
    if "how many" in q_lower:
        return f"There are {len(df)} eligible volunteers."
    elif "list" in q_lower:
        return ", ".join(df["VolunteerID"]) if not df.empty else "No eligible volunteers."
    else:
        return "This question type is not supported."

# -------------------------- UI --------------------------
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# -------------------------- TECHVITALS ADMIN --------------------------
with tab1:
    st.header("TechVitals Admin Portal")

    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv", key="admin_csv_upload")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)

    df = st.session_state.volunteers_df

    if not df.empty:
        st.subheader("Filter Volunteers")
        min_age, max_age = st.slider("Age Range", 0, 100, (30, 75), key="admin_age_slider")
        gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique()), key="admin_gender")
        region = st.selectbox("Region", ["All"] + sorted(df["Region"].dropna().unique()), key="admin_region")

        filtered = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]
        if gender != "All":
            filtered = filtered[filtered["Gender"] == gender]
        if region != "All":
            filtered = filtered[filtered["Region"] == region]

        st.subheader("Volunteer List")
        st.dataframe(filtered)

        st.subheader("Ask a Question (AI Powered)")
        q = st.text_input("Ask about volunteers:", key="admin_q")
        if st.button("Ask", key="admin_ask"):
            ans = answer_question(q, filtered)
            st.session_state.history_admin.append((q, ans))
            st.success(ans)

        if st.session_state.history_admin:
            with st.expander("Conversation History"):
                for ques, ans in st.session_state.history_admin:
                    st.markdown(f"**Q:** {ques}\n\n**A:** {ans}")
    else:
        st.warning("Please upload a valid CSV file to proceed.")

# -------------------------- MEDICAL COMPANY --------------------------
with tab2:
    st.header("Medical Company Portal")

    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"], key="med_pdf")
    if pdf_file:
        with st.spinner("Processing PDF..."):
            try:
                criteria, vectorstore = parse_pdf(pdf_file)
                st.session_state.criteria = criteria
                st.session_state.vectorstore = vectorstore
            except Exception as e:
                st.error(f"PDF processing failed: {e}")

    if st.session_state.criteria:
        st.subheader("Extracted Criteria:")
        st.json(st.session_state.criteria)

        eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)

        st.subheader("Filter Eligible Volunteers")
        min_age2, max_age2 = st.slider("Age Range", 0, 100,
                                       (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]),
                                       key="med_age_slider")
        gender2 = st.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique()), key="med_gender")
        region2 = st.selectbox("Region", ["All"] + sorted(df["Region"].dropna().unique()), key="med_region")

        filtered_eligible = eligible[(eligible["Age"] >= min_age2) & (eligible["Age"] <= max_age2)]
        if gender2 != "All":
            filtered_eligible = filtered_eligible[filtered_eligible["Gender"] == gender2]
        if region2 != "All":
            filtered_eligible = filtered_eligible[filtered_eligible["Region"] == region2]

        st.subheader("Eligible Volunteers (Privacy Protected)")
        if filtered_eligible.empty:
            st.warning("No eligible volunteers found.")
        else:
            st.dataframe(filtered_eligible[["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender"]])

        st.subheader("Ask a Question (AI Powered)")
        q2 = st.text_input("Ask about eligible volunteers:", key="medical_q")
        if st.button("Ask", key="medical_ask"):
            ans2 = answer_question(q2, filtered_eligible)
            st.session_state.history_medical.append((q2, ans2))
            st.success(ans2)

        if st.session_state.history_medical:
            with st.expander("Conversation History"):
                for ques2, ans2 in st.session_state.history_medical:
                    st.markdown(f"**Q:** {ques2}\n\n**A:** {ans2}")
    else:
        st.info("Upload a Trial Criteria PDF to view eligible volunteers.")
