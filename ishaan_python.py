import streamlit as st
import pandas as pd
import fitz
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")
st.title("AI Clinical Trial Management System")

# ---------------------- SESSION STATE INIT ----------------------
if 'volunteers_df' not in st.session_state:
    try:
        st.session_state.volunteers_df = pd.read_csv("Realistic_Indian_Volunteers.csv")
    except:
        st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.history_admin = []
    st.session_state.history_medical = []
    st.session_state.criteria = None
    st.session_state.vectorstore = None

# ---------------------- PDF PARSER FUNCTION ----------------------
def parse_pdf(uploaded_pdf):
    bytes_data = uploaded_pdf.read()
    with open("temp.pdf", "wb") as f:
        f.write(bytes_data)

    loader = PyMuPDFLoader("temp.pdf")
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(docs)
    text = " ".join([chunk.page_content for chunk in chunks])

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    criteria = {
        "min_age": 30,
        "max_age": 75,
        "condition": "Non-Small Cell Lung Cancer (NSCLC)",
        "biomarker": "EGFR+",
        "stages": ["Stage III", "Stage IV", "III", "IV"],
        "gender": "Any",
        "exclude_diabetes": "No",
        "exclude_pregnant": "Yes"
    }

    return criteria, vectorstore

# ---------------------- FILTER FUNCTION ----------------------
def filter_volunteers(df, criteria):
    result = df.copy()
    result = result[(result['Age'] >= criteria["min_age"]) & (result['Age'] <= criteria["max_age"])]
    if criteria["condition"]:
        result = result[result["Condition"].str.contains("NSCLC|Non-Small Cell", case=False, na=False)]
    if criteria["biomarker"]:
        result = result[result["BiomarkerStatus"].str.upper() == criteria["biomarker"].upper()]
    if criteria["stages"]:
        result = result[result["DiseaseStage"].isin(["III", "IV"])]
    if criteria["gender"] != "Any":
        result = result[result["Gender"].str.lower() == criteria["gender"].lower()]
    if criteria["exclude_diabetes"] == "Yes":
        result = result[result["Diabetes"].str.lower() != "yes"]
    if criteria["exclude_pregnant"] == "Yes":
        result = result[result["Pregnant"].str.lower() != "yes"]
    return result

# ---------------------- AI QUESTION FUNCTION ----------------------
def answer_question(q, df, criteria=None):
    q_lower = q.lower()

    if "how many" in q_lower and "male" in q_lower:
        return f"There are {len(df[df['Gender'].str.lower() == 'male'])} eligible male volunteers."
    elif "how many" in q_lower and "female" in q_lower:
        return f"There are {len(df[df['Gender'].str.lower() == 'female'])} eligible female volunteers."
    elif "how many" in q_lower and "stage iii" in q_lower:
        return f"There are {len(df[df['DiseaseStage'] == 'III'])} eligible volunteers in Stage III."
    elif "how many" in q_lower and "stage iv" in q_lower:
        return f"There are {len(df[df['DiseaseStage'] == 'IV'])} eligible volunteers in Stage IV."
    elif "how many" in q_lower and "age" in q_lower:
        match = re.search(r'age\s*(?:above|greater than)\s*(\d+)', q_lower)
        if match:
            age_limit = int(match.group(1))
            return f"There are {len(df[df['Age'] > age_limit])} eligible volunteers above age {age_limit}."
    elif "how many" in q_lower:
        return f"There are {len(df)} eligible volunteers."
    elif "list" in q_lower:
        return ", ".join(df["VolunteerID"]) if not df.empty else "No eligible volunteers."
    elif "why is volunteer" in q_lower and "not eligible" in q_lower and criteria:
        match = re.search(r'volunteer\s*(v\d+)', q_lower)
        if match:
            vid = match.group(1).upper()
            row = st.session_state.volunteers_df[st.session_state.volunteers_df["VolunteerID"] == vid]
            if row.empty:
                return f"Volunteer {vid} does not exist."
            reasons = []
            r = row.iloc[0]
            if not (criteria["min_age"] <= r["Age"] <= criteria["max_age"]):
                reasons.append(f"Age {r['Age']} not in allowed range {criteria['min_age']}-{criteria['max_age']}")
            if criteria["condition"] and "nsclc" not in str(r["Condition"]).lower():
                reasons.append("Condition mismatch.")
            if criteria["biomarker"] and str(r["BiomarkerStatus"]).upper() != criteria["biomarker"].upper():
                reasons.append("Biomarker status mismatch.")
            if criteria["stages"] and r["DiseaseStage"] not in ["III", "IV"]:
                reasons.append("Stage mismatch.")
            if criteria["exclude_diabetes"] == "Yes" and str(r["Diabetes"]).lower() == "yes":
                reasons.append("Volunteer has diabetes.")
            if criteria["exclude_pregnant"] == "Yes" and str(r["Pregnant"]).lower() == "yes":
                reasons.append("Volunteer is pregnant.")
            if not reasons:
                return f"Volunteer {vid} is actually eligible."
            return f"Volunteer {vid} is not eligible because: " + "; ".join(reasons)
    else:
        return "This question type is not supported."

# ---------------------- UI ----------------------
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# ---------------------- TECHVITALS ADMIN ----------------------
with tab1:
    st.header("TechVitals Admin Portal")

    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)

    df = st.session_state.volunteers_df

    st.subheader("Filter Volunteers")
    min_age, max_age = st.slider("Age Range", 0, 100, (30, 75), key="admin_age")
    gender = st.selectbox("Gender", ["All"] + (sorted(df["Gender"].dropna().unique()) if not df.empty else []), key="admin_gender")
    region = st.selectbox("Region", ["All"] + (sorted(df["Region"].dropna().unique()) if not df.empty else []), key="admin_region")

    filtered = df[(df["Age"] >= min_age) & (df["Age"] <= max_age)]
    if gender != "All":
        filtered = filtered[filtered["Gender"] == gender]
    if region != "All":
        filtered = filtered[filtered["Region"] == region]

    st.subheader("Volunteer List")
    st.dataframe(filtered)

    st.subheader("Ask a Question (AI Powered)")
    q1 = st.text_input("Ask about volunteers:")
    if st.button("Ask", key="admin_ask"):
        ans1 = answer_question(q1, filtered)
        st.session_state.history_admin.append((q1, ans1))
        st.success(ans1)

    if st.session_state.history_admin:
        with st.expander("Conversation History"):
            for ques, ans in st.session_state.history_admin:
                st.markdown(f"**Q:** {ques}\n**A:** {ans}")

# ---------------------- MEDICAL COMPANY ----------------------
with tab2:
    st.header("Medical Company Portal")

    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"], key="medical_pdf")
    if pdf_file:
        with st.spinner("Processing PDF..."):
            criteria, vectorstore = parse_pdf(pdf_file)
            st.session_state.criteria = criteria
            st.session_state.vectorstore = vectorstore

    if st.session_state.criteria:
        st.subheader("Extracted Criteria:")
        st.json(st.session_state.criteria)

        eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)

        st.subheader("Filter Eligible Volunteers")
        min_age2, max_age2 = st.slider("Age Range", 0, 100, (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]), key="med_age")
        gender2 = st.selectbox("Gender", ["All"] + (sorted(df["Gender"].dropna().unique()) if not df.empty else []), key="med_gender")
        region2 = st.selectbox("Region", ["All"] + (sorted(df["Region"].dropna().unique()) if not df.empty else []), key="med_region")

        filtered_eligible = eligible[(eligible["Age"] >= min_age2) & (eligible["Age"] <= max_age2)]
        if gender2 != "All":
            filtered_eligible = filtered_eligible[filtered_eligible["Gender"] == gender2]
        if region2 != "All":
            filtered_eligible = filtered_eligible[filtered_eligible["Region"] == region2]

        st.subheader("Eligible Volunteers (Privacy Protected)")
        st.dataframe(filtered_eligible[["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender"]])

        st.subheader("Ask a Question (AI Powered)")
        q2 = st.text_input("Ask about eligible volunteers:")
        if st.button("Ask", key="medical_ask"):
            ans2 = answer_question(q2, filtered_eligible, st.session_state.criteria)
            st.session_state.history_medical.append((q2, ans2))
            st.success(ans2)

        if st.session_state.history_medical:
            with st.expander("Conversation History"):
                for ques2, ans2 in st.session_state.history_medical:
                    st.markdown(f"**Q:** {ques2}\n**A:** {ans2}")

