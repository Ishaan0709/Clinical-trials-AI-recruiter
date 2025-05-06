import streamlit as st
import pandas as pd
import fitz
import re
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")
st.title("AI Clinical Trial Management System")

# -------------------------- SESSION STATE INIT --------------------------
if 'volunteers_df' not in st.session_state:
    st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.history_admin = []
    st.session_state.history_medical = []
    st.session_state.criteria = None
    st.session_state.vectorstore = None

# -------------------------- PDF PARSER FUNCTION --------------------------
def parse_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    criteria = {
        "min_age": 30,
        "max_age": 75,
        "condition": "NSCLC",
        "biomarker": "EGFR+",
        "stages": ["III", "IV"],
        "gender": "Any",
        "exclude_diabetes": "No",
        "exclude_pregnant": "Yes"
    }

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

# -------------------------- ENTITY EXTRACTION --------------------------
def extract_entities(q):
    genders = []
    stages = []
    regions = []
    biomarkers = []
    ages = []
    conditions = []
    q_lower = q.lower()

    # Gender detection
    if "male" in q_lower:
        genders.append("Male")
    if "female" in q_lower:
        genders.append("Female")

    # Stage detection
    for s in ["I", "II", "III", "IV"]:
        if f"stage {s}".lower() in q_lower:
            stages.append(s)

    # Biomarker detection
    for b in ["EGFR+", "ALK+", "KRAS+", "ROS1+"]:
        if b.lower() in q_lower:
            biomarkers.append(b)

    # Region detection
    region_keywords = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad"]
    for r in region_keywords:
        if r.lower() in q_lower:
            regions.append(r)

    # Age detection
    age_numbers = re.findall(r'(?:above|over|greater than|older than|age)\s*(\d+)', q_lower)
    if age_numbers:
        ages.append(int(age_numbers[0]))

    # Condition detection
    if "nsclc" in q_lower or "non-small cell" in q_lower:
        conditions.append("NSCLC")

    return {
        "gender": genders,
        "stage": stages,
        "region": regions,
        "biomarker": biomarkers,
        "age": ages,
        "condition": conditions
    }

# -------------------------- ENHANCED QA FUNCTIONS --------------------------
def medical_gpt_answer(q, df, criteria):
    llm = ChatOpenAI(temperature=0)
    prompt_template = f"""
    MEDICAL TRIAL CRITERIA:
    {criteria if criteria else 'No specific criteria provided'}
    
    VOLUNTEER DATA:
    {df[["VolunteerID", "Age", "Gender", "Condition", "BiomarkerStatus", "DiseaseStage"]].to_string(index=False)}
    
    QUESTION: {q}
    
    RESPONSE RULES:
    1. Always reference criteria when discussing eligibility
    2. Never list more than 5 volunteer IDs
    3. Explain biomarker and stage requirements
    4. Keep responses under 3 sentences unless complex analysis needed
    5. Suggest filters when appropriate
    """
    return llm.predict(prompt_template)

def answer_question(q, df, criteria=None):
    q_lower = q.lower()
    
    # Handle greetings first
    if any(word in q_lower for word in ["hello", "hi", "hey"]):
        return "Hello! I'm your clinical trial assistant. Ask me about eligibility criteria, volunteer demographics, or trial requirements."
    
    # Medical panel specific logic
    if criteria:
        # Eligibility explanations
        if "why is volunteer" in q_lower:
            vid = re.findall(r'volunteer\s*([vV]\d+)', q_lower)
            if vid:
                vid = vid[0].upper()
                full_df = st.session_state.volunteers_df
                row = full_df[full_df["VolunteerID"].str.upper() == vid]
                
                if row.empty:
                    return f"Volunteer {vid} not found in dataset."
                
                if vid in df["VolunteerID"].values:
                    return f"Volunteer {vid} meets all eligibility criteria."
                else:
                    reasons = []
                    r = row.iloc[0]
                    if r["Age"] < criteria["min_age"] or r["Age"] > criteria["max_age"]:
                        reasons.append(f"Age {r['Age']} outside range {criteria['min_age']}-{criteria['max_age']}")
                    if criteria["condition"] and not re.search(criteria["condition"], str(r["Condition"]), re.I):
                        reasons.append(f"Condition '{r['Condition']}' doesn't match '{criteria['condition']}'")
                    if criteria["biomarker"] and str(r["BiomarkerStatus"]).upper() != criteria["biomarker"].upper():
                        reasons.append(f"Biomarker '{r['BiomarkerStatus']}' ≠ '{criteria['biomarker']}'")
                    if criteria["stages"] and r["DiseaseStage"] not in criteria["stages"]:
                        reasons.append(f"Stage '{r['DiseaseStage']}' not in {criteria['stages']}")
                    if criteria["gender"] != "Any" and r["Gender"].lower() != criteria["gender"].lower():
                        reasons.append(f"Gender '{r['Gender']}' ≠ '{criteria['gender']}'")
                    if criteria["exclude_diabetes"] == "Yes" and str(r["Diabetes"]).lower() == "yes":
                        reasons.append("Has diabetes")
                    if criteria["exclude_pregnant"] == "Yes" and str(r["Pregnant"]).lower() == "yes":
                        reasons.append("Pregnant")
                    
                    return f"Volunteer {vid} ineligible: " + ", ".join(reasons) if reasons else f"Volunteer {vid} doesn't meet hidden criteria"
        
        # Prevent raw listings
        if "list" in q_lower or "show" in q_lower:
            clean_df = df.drop_duplicates("VolunteerID")
            if len(clean_df) > 10:
                return f"Found {len(clean_df)} volunteers. Try adding filters like:\n- 'Show females under 50'\n- 'EGFR+ patients from Delhi'"
            return ", ".join(clean_df["VolunteerID"].tolist())
        
        # All other medical questions
        return medical_gpt_answer(q, df, criteria)
    
    # Admin panel logic
    else:
        # ... (keep existing admin panel answer_question logic)
        
# -------------------------- UI --------------------------
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# -------------------------- ADMIN PANEL --------------------------
with tab1:
    st.header("TechVitals Admin Portal")
    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)
        st.success(f"Dataset uploaded with {len(st.session_state.volunteers_df)} volunteers!")

    df = st.session_state.volunteers_df

    if not df.empty:
        # ... (keep existing admin panel UI code)

# -------------------------- MEDICAL COMPANY PANEL --------------------------
with tab2:
    st.header("Medical Company Portal")
    
    if st.session_state.volunteers_df.empty:
        st.warning("Waiting for TechVitals dataset...")
    else:
        # Enhanced medical summary
        summary_data = {
            "total": len(st.session_state.volunteers_df),
            "age_min": st.session_state.volunteers_df['Age'].min(),
            "age_max": st.session_state.volunteers_df['Age'].max(),
            "top_condition": st.session_state.volunteers_df['Condition'].mode()[0],
            "gender_dist": st.session_state.volunteers_df['Gender'].value_counts().to_dict()
        }
        
        st.markdown(f"""
        **Medical Trial Overview**
        - Total Volunteers: {summary_data['total']}
        - Age Range: {summary_data['age_min']}-{summary_data['age_max']}
        - Most Common Condition: {summary_data['top_condition']}
        - Gender Distribution: {summary_data['gender_dist']}
        """)

        # PDF processing
        pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
        if pdf_file:
            with st.spinner("Analyzing criteria..."):
                criteria, vectorstore = parse_pdf(pdf_file)
                st.session_state.criteria = criteria
                st.session_state.vectorstore = vectorstore
                st.success("Medical criteria loaded!")

        if st.session_state.criteria:
            # ... (keep existing medical panel UI code with added deduplication)
            
            st.subheader("Ask Trial Questions")
            q2 = st.text_input("Ask about eligibility:", key="medical_q")
            if st.button("Ask Medical Expert", key="medical_ask"):
                eligible_df = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)
                ans2 = answer_question(q2, eligible_df, st.session_state.criteria)
                
                st.session_state.history_medical.append((q2, ans2))
                st.markdown(f"**Answer:**\n{ans2}")

            if st.session_state.history_medical:
                with st.expander("Conversation History"):
                    for ques, ans in st.session_state.history_medical:
                        st.markdown(f"**Q:** {ques}\n\n**A:** {ans}")
