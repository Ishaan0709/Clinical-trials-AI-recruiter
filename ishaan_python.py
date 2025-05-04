import streamlit as st
import pandas as pd
import fitz
import re
import tempfile
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")
st.title("AI Clinical Trial Management System")

# -------------------------- SESSION STATE INIT --------------------------
if 'volunteers_df' not in st.session_state:
    try:
        st.session_state.volunteers_df = pd.read_csv("Realistic_Indian_Volunteers.csv")
    except:
        st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.history_admin = []
    st.session_state.history_medical = []
    st.session_state.criteria = None
    st.session_state.vectorstore = None

# -------------------------- PDF PARSER FUNCTION --------------------------
def parse_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.read())
        tmp_file_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    text = " ".join([chunk.page_content for chunk in chunks])

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

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

    if re.search("Non-Small Cell Lung Cancer|NSCLC", text):
        criteria["condition"] = "Non-Small Cell Lung Cancer (NSCLC)"

    if "EGFR+" in text:
        criteria["biomarker"] = "EGFR+"

    stage_match = re.findall(r'\b(I{1,3}|IV)\b', text)
    if stage_match:
        criteria["stages"] = list(set(stage_match))

    if "pregnant" in text.lower():
        criteria["exclude_pregnant"] = "Yes"

    return criteria, vectorstore

# -------------------------- FILTER FUNCTION --------------------------
def filter_volunteers(df, criteria):
    result = df.copy()
    result = result[(result['Age'] >= criteria["min_age"]) & (result['Age'] <= criteria["max_age"])]
    if criteria["condition"]:
        result = result[result["Condition"].str.contains("NSCLC|Non-Small Cell", case=False, na=False)]
    if criteria["biomarker"]:
        result = result[result["BiomarkerStatus"].str.upper() == criteria["biomarker"].upper()]
    if criteria["stages"]:
        result = result[result["DiseaseStage"].isin(criteria["stages"])]
    if criteria["gender"] != "Any":
        result = result[result["Gender"].str.lower() == criteria["gender"].lower()]
    if criteria["exclude_diabetes"] == "Yes":
        result = result[result["Diabetes"].str.lower() != "yes"]
    if criteria["exclude_pregnant"] == "Yes":
        result = result[result["Pregnant"].str.lower() != "yes"]
    return result

# -------------------------- QA FUNCTION --------------------------
def answer_question(q, df, criteria=None):
    q_lower = q.lower()

    # Count questions
    if "how many" in q_lower:
        if "stage iii" in q_lower:
            count = len(df[df["DiseaseStage"] == "III"])
            return f"There are {count} eligible volunteers."
        if "stage iv" in q_lower:
            count = len(df[df["DiseaseStage"] == "IV"])
            return f"There are {count} eligible volunteers."
        if "male" in q_lower:
            count = len(df[df["Gender"].str.lower() == "male"])
            return f"There are {count} eligible volunteers."
        if "female" in q_lower:
            count = len(df[df["Gender"].str.lower() == "female"])
            return f"There are {count} eligible volunteers."
        if "age" in q_lower and "above" in q_lower:
            age_val = int(re.findall(r'\d+', q_lower)[0])
            count = len(df[df["Age"] > age_val])
            return f"There are {count} eligible volunteers."
        return f"There are {len(df)} eligible volunteers."

    # List questions
    elif "list" in q_lower:
        if not df.empty:
            return ", ".join(df["VolunteerID"])
        else:
            return "No eligible volunteers."

    # Why not eligible
    elif "why" in q_lower and "not eligible" in q_lower and "volunteer" in q_lower:
        vid_match = re.findall(r'volunteer\s*(\w+)', q_lower)
        if vid_match and criteria is not None:
            vid = vid_match[0].upper()
            full_df = st.session_state.volunteers_df
            volunteer = full_df[full_df["VolunteerID"].str.upper() == vid]
            if volunteer.empty:
                return f"Volunteer {vid} not found in dataset."

            v = volunteer.iloc[0]
            reasons = []
            if v["Age"] < criteria["min_age"] or v["Age"] > criteria["max_age"]:
                reasons.append("Age not in required range.")
            if criteria["condition"] and ("NSCLC" not in str(v["Condition"])):
                reasons.append("Condition does not match.")
            if criteria["biomarker"] and (str(v["BiomarkerStatus"]).upper() != criteria["biomarker"].upper()):
                reasons.append("Biomarker status does not match.")
            if criteria["stages"] and v["DiseaseStage"] not in criteria["stages"]:
                reasons.append("Disease stage does not match.")
            if criteria["gender"] != "Any" and v["Gender"].lower() != criteria["gender"].lower():
                reasons.append("Gender does not match.")
            if criteria["exclude_diabetes"] == "Yes" and str(v["Diabetes"]).lower() == "yes":
                reasons.append("Volunteer has diabetes.")
            if criteria["exclude_pregnant"] == "Yes" and str(v["Pregnant"]).lower() == "yes":
                reasons.append("Volunteer is pregnant.")
            return " | ".join(reasons) if reasons else "Volunteer is eligible."
        else:
            return "Please specify a valid Volunteer ID."

    else:
        return "This question type is not supported."

# -------------------------- UI --------------------------

tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# -------------------------- TECHVITALS ADMIN --------------------------
with tab1:
    st.header("TechVitals Admin Portal")

    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)

    df = st.session_state.volunteers_df

    # Filters
    st.subheader("Filter Volunteers")
    min_age, max_age = st.slider("Age Range", 0, 100, (30, 75), key="admin_age")
    gender = st.selectbox("Gender", ["All"] + sorted(df["Gender"].dropna().unique()), key="admin_gender")
    region = st.selectbox("Region", ["All"] + sorted(df["Region"].dropna().unique()), key="admin_region")

    filtered = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]
    if gender != "All":
        filtered = filtered[filtered["Gender"] == gender]
    if region != "All":
        filtered = filtered[filtered["Region"] == region]

    st.subheader("Volunteer List")
    st.dataframe(filtered)

    # Q&A
    st.subheader("Ask a Question (AI Powered)")
    q = st.text_input("Ask about volunteers:")
    if st.button("Ask", key="admin_ask"):
        ans = answer_question(q, filtered)
        st.session_state.history_admin.append((q, ans))
        st.success(ans)

    if st.session_state.history_admin:
        with st.expander("Conversation History"):
            for ques, ans in st.session_state.history_admin:
                st.markdown(f"**Q:** {ques}\n\n**A:** {ans}")

# -------------------------- MEDICAL COMPANY --------------------------
with tab2:
    st.header("Medical Company Portal")

    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF..."):
            criteria, vectorstore = parse_pdf(pdf_file)
            st.session_state.criteria = criteria
            st.session_state.vectorstore = vectorstore

    if st.session_state.criteria:
        st.subheader("Extracted Criteria:")
        st.json(st.session_state.criteria)

        eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)

        # Filters
        min_age2, max_age2 = st.slider("Age Range", 0, 100, (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]), key="med_age")
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
            st.dataframe(filtered_eligible[["Volunteer

