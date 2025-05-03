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
    raise ValueError("API key missing ya invalid hai.")

model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
embeddings = OpenAIEmbeddings()

# ------------------- VOLUNTEER MATCHING LOGIC -----------------------
def match_volunteers(df, trial_info):
    eligible = []
    min_age, max_age = trial_info.get("age_range", (0, 120))
    required_condition = trial_info.get("condition", "").lower()
    gender_req = trial_info.get("gender", [])
    exclude_diabetes = trial_info.get("exclude_diabetes", False)
    exclude_pregnant = trial_info.get("exclude_pregnant", False)

    for _, row in df.iterrows():
        age = row["Age"]
        cond = str(row["Condition"]).lower()
        gender = row["Gender"]

        if min_age <= age <= max_age and required_condition in cond:
            if not gender_req or gender in gender_req or "Any" in gender_req:
                if (not exclude_diabetes or row["Diabetes"] == "No") and \
                   (not exclude_pregnant or row["Pregnant"] == "No"):
                    eligible.append(row)
    return pd.DataFrame(eligible)

# ------------------- PDF TO TRIAL CRITERIA -----------------------
def extract_criteria_from_pdf(pdf_file):
    loader = PyMuPDFLoader(pdf_file)
    docs = loader.load()
    context = "\n".join(doc.page_content for doc in docs)

    prompt = f"""
    Extract the following trial criteria:
    {{
    "drug_name": str,
    "condition": str,
    "age_range": [min_age, max_age],
    "biomarker": str/null,
    "gender": list,
    "stages": list,
    "exclude_diabetes": bool,
    "exclude_pregnant": bool
    }}
    Text: {context}
    """
    response = model.invoke(prompt)
    try:
        return json.loads(response.content.strip())
    except:
        return {}

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

# ------------------- LOAD DOCUMENTS FOR Q&A -----------------------
def load_documents(df):
    text = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
    doc = Document(page_content=text)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents([doc])
    return FAISS.from_documents(chunks, embeddings)

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
        st.dataframe(df)
        st.session_state.volunteer_df = df

        # FILTERS
        df_filtered = display_filters(df)
        st.dataframe(df_filtered)

        # Q&A
        st.markdown("### 💬 Ask a question about the dataset:")
        question = st.text_input("E.g. List all volunteers above 50 age")

        if question:
            vector = load_documents(df)
            result = vector.similarity_search(question, k=4)
            context = "\n\n".join([doc.page_content for doc in result])
            answer = model.invoke(f"Data:\n{context}\n\nQuestion: {question}\nAnswer in simple bullet points:")
            st.markdown("**Answer:** " + answer.content)

            st.session_state.query_history.append(question)

            with st.expander("📜 Previous Questions"):
                for q in st.session_state.query_history:
                    st.markdown("- " + q)

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
                    st.dataframe(filtered_matches[["VolunteerID", "Name", "Email", "Age", "Gender"]])
                else:
                    st.warning("No matching volunteers found for this trial.")

                # QUESTION BOX FOR ELIGIBLE VOLUNTEERS
                st.markdown("### 🔎 Ask about eligible volunteers:")
                question = st.text_input("Your question", key="eligible_qa")

                if question and not matches.empty:
                    vector = load_documents(matches)
                    result = vector.similarity_search(question, k=4)
                    context = "\n\n".join([doc.page_content for doc in result])
                    answer = model.invoke(f"Data:\n{context}\n\nQuestion: {question}\nAnswer in bullet points:")
                    st.markdown("**Answer:** " + answer.content)

        elif mode == "Fill Live Trial Form":
            with st.form("live_trial"):
                drug = st.text_input("Drug Name")
                condition = st.text_input("Condition")
                age_range = st.slider("Age Range", 18, 90, (30, 70))
                gender = st.multiselect("Allowed Gender", ["Male", "Female", "Other"])
                exclude_diabetes = st.checkbox("Exclude Diabetic Volunteers")
                exclude_pregnant = st.checkbox("Exclude Pregnant Volunteers")
                submitted = st.form_submit_button("Submit Trial")

            if submitted:
                df = st.session_state.volunteer_df
                matches = match_volunteers(df, {
                    "age_range": age_range,
                    "condition": condition,
                    "gender": gender,
                    "exclude_diabetes": exclude_diabetes,
                    "exclude_pregnant": exclude_pregnant
                })
                if not matches.empty:
                    st.success(f"🎯 {len(matches)} volunteers matched this trial!")
                    filtered_matches = display_filters(matches)
                    st.dataframe(filtered_matches[["VolunteerID", "Name", "Email", "Age", "Gender"]])
                else:
                    st.warning("No eligible volunteers found.")

