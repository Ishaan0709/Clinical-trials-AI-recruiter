import streamlit as st
import pandas as pd
import os
import tempfile
import json
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

# ---------------- API KEY SETUP ---------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
    raise ValueError("API key missing or invalid")

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# ------------------- ENHANCED VOLUNTEER MATCHING -----------------------
def match_volunteers(df, trial_info):
    eligible = []
    min_age, max_age = trial_info.get("age_range", (0, 120))
    required_condition = trial_info.get("condition", "").lower()
    required_biomarker = trial_info.get("biomarker", "").upper().replace(" ", "")
    required_stages = [s.upper().strip() for s in trial_info.get("stages", [])]
    gender_req = [g.strip() for g in trial_info.get("gender", [])]

    for _, row in df.iterrows():
        # Normalize data
        volunteer_condition = str(row["Condition"]).lower()
        volunteer_biomarker = str(row.get("BiomarkerStatus", "")).upper().replace(" ", "")
        volunteer_stage = str(row.get("DiseaseStage", "")).upper().strip()
        volunteer_gender = str(row["Gender"]).strip()

        # Condition matching with synonyms
        condition_match = any([
            required_condition in volunteer_condition,
            "nsclc" in required_condition and "non-small" in volunteer_condition,
            "egfr" in required_condition.lower() and "egfr" in volunteer_biomarker
        ])
        
        # Stage matching with Roman numerals
        stage_match = (not required_stages) or any(
            stage in volunteer_stage 
            for stage in required_stages
        )
        
        # Biomarker matching
        biomarker_match = (not required_biomarker) or required_biomarker in volunteer_biomarker
        
        # Gender matching
        gender_match = (not gender_req) or (volunteer_gender in gender_req) or ("Any" in gender_req)
        
        # Medical restrictions
        diabetes_ok = (not trial_info.get("exclude_diabetes", False)) or (row["Diabetes"] == "No")
        pregnant_ok = (not trial_info.get("exclude_pregnant", False)) or (row["Pregnant"] == "No")

        if all([
            min_age <= row["Age"] <= max_age,
            condition_match,
            stage_match,
            biomarker_match,
            gender_match,
            diabetes_ok,
            pregnant_ok
        ]):
            eligible.append(row)
            
    return pd.DataFrame(eligible)

# ------------------- IMPROVED PDF CRITERIA EXTRACTION -----------------------
def extract_criteria_from_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    docs = loader.load()
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""Extract EXACT trial criteria from this document:
    {{
    "drug_name": "Standard drug name (fix typos)",
    "condition": "Full medical condition with biomarkers",
    "age_range": [min_age, max_age],
    "biomarker": "Specific biomarker required",
    "gender": ["Allowed genders"],
    "stages": ["Roman numeral stages"],
    "exclude_diabetes": boolean,
    "exclude_pregnant": boolean
    }}

    Rules:
    1. Convert stages to Roman numerals (I, II, III, IV)
    2. Normalize biomarkers (EGFR+ → EGFR positive)
    3. Expand cancer abbreviations (NSCLC → Non-Small Cell Lung Cancer)
    4. Fix drug name typos

    Document content: {context}
    """
    
    try:
        response = model.invoke(prompt)
        criteria = json.loads(response.content.strip())
        # Normalize stages to uppercase Roman
        if "stages" in criteria:
            criteria["stages"] = [s.upper().replace("STAGE", "").strip() for s in criteria["stages"]]
        return criteria
    except Exception as e:
        st.error(f"Failed to parse criteria: {e}")
        return {}

# ------------------- ENHANCED Q&A CONTEXT -----------------------
def load_documents(df):
    text = []
    for _, row in df.iterrows():
        entry = (
            f"Volunteer {row['VolunteerID']}: "
            f"{row['Age']}yo {row['Gender']} with {row['Condition']} "
            f"(Stage {row['DiseaseStage']}, Biomarker: {row['BiomarkerStatus']}) "
            f"{'Diabetic' if row['Diabetes'] == 'Yes' else ''} "
            f"{'Pregnant' if row['Pregnant'] == 'Yes' else ''}"
        )
        text.append(entry)
    
    doc = Document(page_content="\n".join(text))
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    return FAISS.from_documents(chunks, embeddings)

# ------------------- FILTER WIDGETS -----------------------
def display_filters(df):
    st.markdown("### 🔎 Filter Volunteers")
    gender = st.selectbox("Gender", ["All"] + df["Gender"].unique().tolist())
    age_range = st.slider("Age Range", 18, 90, (30, 70))
    region = st.selectbox("Region", ["All"] + sorted(df["Region"].unique())) if "Region" in df.columns else "All"

    if gender != "All":
        df = df[df["Gender"] == gender]
    df = df[df["Age"].between(age_range[0], age_range[1])]
    if region != "All":
        df = df[df["Region"] == region]
    return df

# ------------------- APP -----------------------
st.set_page_config(page_title="Clinical Trial AI Recruiter", layout="wide")
st.title("🤖 AI Clinical Trial Volunteer Recruiter")

tab1, tab2 = st.tabs(["👨‍💼 TechVitals Admin", "🏥 Medical Company"])

# ------------ SESSION STATE INIT -----------
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# =========== TECHVITALS ADMIN ============
with tab1:
    st.header("Upload Volunteer Dataset")
    csv_file = st.file_uploader("Upload CSV", type=["csv"])

    if csv_file:
        df = pd.read_csv(csv_file)
        st.success("✅ Dataset uploaded successfully!")
        st.dataframe(df.head())
        st.session_state.volunteer_df = df

        # FILTERS
        df_filtered = display_filters(df)
        st.dataframe(df_filtered)

        # Q&A
        st.markdown("### 💬 Ask a question about the dataset:")
        question = st.text_input("E.g. List all volunteers above 50 with EGFR+ biomarker")

        if question:
            vector = load_documents(df)
            result = vector.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in result])
            answer = model.invoke(f"Data:\n{context}\n\nQuestion: {question}\nAnswer in bullet points:")
            st.markdown("**Answer:** " + answer.content)
            st.session_state.query_history.append(question)

            with st.expander("📜 Previous Questions"):
                for q in st.session_state.query_history[-5:]:
                    st.markdown(f"- {q}")

# =========== MEDICAL COMPANY ============
with tab2:
    st.header("Medical Company Trial Submission")

    if "volunteer_df" not in st.session_state:
        st.warning("Please upload volunteer data in Admin tab first.")
    else:
        mode = st.radio("Choose input mode:", ["Upload Trial PDF", "Fill Live Trial Form"])

        if mode == "Upload Trial PDF":
            pdf_file = st.file_uploader("Upload Trial PDF", type="pdf")
            if pdf_file:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(pdf_file.read())
                    tmp_path = tmp.name

                trial_criteria = extract_criteria_from_pdf(tmp_path)
                st.success("✅ Extracted Trial Criteria")
                st.json(trial_criteria)

                df = st.session_state.volunteer_df
                matches = match_volunteers(df, trial_criteria)

                if not matches.empty:
                    st.success(f"🎯 Found {len(matches)} eligible volunteers!")
                    filtered_matches = display_filters(matches)
                    cols = ["VolunteerID", "Name", "Age", "Gender", "Condition", 
                           "DiseaseStage", "BiomarkerStatus"]
                    st.dataframe(filtered_matches[cols])
                else:
                    st.warning("No matching volunteers found for this trial.")

                # QUESTION BOX
                st.markdown("### 🔍 Ask about eligible volunteers:")
                question = st.text_input("Your question", key="eligible_qa")
                if question and not matches.empty:
                    vector = load_documents(matches)
                    result = vector.similarity_search(question, k=4)
                    context = "\n\n".join([doc.page_content for doc in result])
                    answer = model.invoke(f"Data:\n{context}\n\nQuestion: {question}\nAnswer:")
                    st.markdown("**Answer:** " + answer.content)

        elif mode == "Fill Live Trial Form":
            with st.form("live_trial"):
                drug = st.text_input("Drug Name")
                condition = st.text_input("Condition", 
                    help="E.g: 'EGFR-positive NSCLC'")
                age_range = st.slider("Age Range", 18, 90, (40, 75))
                gender = st.multiselect("Allowed Gender", 
                    ["Male", "Female", "Other", "Any"])
                biomarker = st.text_input("Required Biomarker", 
                    placeholder="EGFR+")
                stages = st.multiselect("Disease Stages", 
                    ["I", "II", "III", "IV"])
                exclude_diabetes = st.checkbox("Exclude Diabetic Volunteers")
                exclude_pregnant = st.checkbox("Exclude Pregnant Volunteers")
                submitted = st.form_submit_button("Submit Trial")

            if submitted:
                df = st.session_state.volunteer_df
                matches = match_volunteers(df, {
                    "age_range": age_range,
                    "condition": condition,
                    "gender": gender,
                    "biomarker": biomarker,
                    "stages": stages,
                    "exclude_diabetes": exclude_diabetes,
                    "exclude_pregnant": exclude_pregnant
                })
                
                if not matches.empty:
                    st.success(f"🎯 {len(matches)} volunteers matched!")
                    cols = ["VolunteerID", "Name", "Age", "Gender", 
                           "Condition", "DiseaseStage", "BiomarkerStatus"]
                    st.dataframe(matches[cols])
                else:
                    st.warning("No eligible volunteers found.")
