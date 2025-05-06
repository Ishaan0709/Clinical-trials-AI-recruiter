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

# -------------------------- UPDATED PDF PARSER FUNCTION --------------------------
def parse_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    # Extract criteria using regex patterns
    criteria = {
        "min_age": None,
        "max_age": None,
        "condition": None,
        "biomarker": None,
        "stages": [],
        "gender": "Any",
        "exclude_diabetes": "No",
        "exclude_pregnant": "Yes"  # Default to safe option
    }

    # Age extraction
    age_match = re.search(r'Age between (\d+) and (\d+) years', text)
    if age_match:
        criteria["min_age"] = int(age_match.group(1))
        criteria["max_age"] = int(age_match.group(2))

    # Condition extraction
    if 'Non-Small Cell Lung Cancer (NSCLC)' in text:
        criteria["condition"] = "NSCLC"

    # Biomarker extraction
    biomarker_match = re.search(r'EGFR-positive \(EGFR\+\)', text)
    if biomarker_match:
        criteria["biomarker"] = "EGFR+"

    # Stage extraction
    stage_match = re.search(r'Stage (III) or (IV)', text)
    if stage_match:
        criteria["stages"] = [stage_match.group(1), stage_match.group(2)]

    # Exclusion criteria
    if 'Pregnant volunteers excluded' in text:
        criteria["exclude_pregnant"] = "Yes"
    if 'Volunteers with known diabetes not excluded' in text:
        criteria["exclude_diabetes"] = "No"

    # Validate all required fields
    required_fields = ['min_age', 'max_age', 'condition', 'biomarker', 'stages']
    missing_fields = [field for field in required_fields if not criteria[field]]
    
    if missing_fields:
        raise ValueError(f"Invalid PDF format. Missing: {', '.join(missing_fields)}")

    # Create vectorstore
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

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

# -------------------------- RULE-BASED QA FUNCTION --------------------------
def answer_question(q, df, criteria=None):
    q_lower = q.lower()

    # ========== MEDICAL PANEL IMPROVEMENTS START ==========
    if criteria:  # Medical company logic
        # Handle greetings
        if any(word in q_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your medical trial assistant. Ask me about eligibility criteria or volunteer demographics."
        
        # Prevent raw ID listings
        if "list" in q_lower or "show" in q_lower:
            clean_df = df.drop_duplicates("VolunteerID")
            if len(clean_df) > 15:
                return f"Found {len(clean_df)} eligible volunteers. Try adding filters like:\n- 'Female volunteers under 50'\n- 'EGFR+ patients from Mumbai'"
            return ", ".join(clean_df["VolunteerID"].tolist())
        
        # GPT-powered answers for medical panel
        llm = ChatOpenAI(temperature=0)
        prompt = f"""
        MEDICAL CRITERIA:
        {criteria}
        
        VOLUNTEER DATA:
        {df[["VolunteerID", "Age", "Gender", "Condition", "BiomarkerStatus"]].to_string()}
        
        QUESTION: {q}
        
        ANSWER RULES:
        1. Reference criteria when discussing eligibility
        2. Keep responses under 3 sentences
        3. Use exact numbers from data
        4. Never list more than 5 IDs
        """
        return llm.predict(prompt)
    # ========== MEDICAL PANEL IMPROVEMENTS END ==========

    # Original admin panel logic below
    if "how many" in q_lower:
        if "female" in q_lower:
            return f"There are {len(df[df['Gender'].str.lower() == 'female'])} eligible female volunteers."
        if "male" in q_lower:
            return f"There are {len(df[df['Gender'].str.lower() == 'male'])} eligible male volunteers."
        if "diabetes" in q_lower:
            return f"There are {len(df[df['Diabetes'].str.lower() == 'yes'])} volunteers with diabetes."
        if "pregnant" in q_lower:
            return f"There are {len(df[df['Pregnant'].str.lower() == 'yes'])} pregnant volunteers."
        return f"There are {len(df)} eligible volunteers."

    elif "list" in q_lower or "show" in q_lower:
        return ", ".join(df["VolunteerID"]) if not df.empty else "No eligible volunteers."

    elif "why is volunteer" in q_lower and "not eligible" in q_lower:
        vid = re.findall(r'volunteer\s*([vV]\d+)', q_lower)
        if vid:
            vid = vid[0].upper()
            full_df = st.session_state.volunteers_df
            row = full_df[full_df["VolunteerID"].str.upper() == vid]

            if row.empty:
                return f"Volunteer ID {vid} not found in the dataset."

            if vid in df["VolunteerID"].str.upper().values:
                return f"Sir/Madam, the volunteer {vid} is fully eligible for the clinical trial."

            else:
                reasons = []
                r = row.iloc[0]
                if r["Age"] < criteria["min_age"] or r["Age"] > criteria["max_age"]:
                    reasons.append("Age not in eligible range.")
                if criteria["condition"] and criteria["condition"].lower() not in str(r["Condition"]).lower():
                    reasons.append("Condition does not match.")
                if criteria["biomarker"] and str(r["BiomarkerStatus"]).upper() != criteria["biomarker"].upper():
                    reasons.append("Biomarker does not match.")
                if criteria["stages"] and r["DiseaseStage"] not in criteria["stages"]:
                    reasons.append("Disease stage does not match.")
                if criteria["gender"] != "Any" and r["Gender"].lower() != criteria["gender"].lower():
                    reasons.append("Gender does not match.")
                if criteria["exclude_diabetes"] == "Yes" and str(r["Diabetes"]).lower() == "yes":
                    reasons.append("Has diabetes.")
                if criteria["exclude_pregnant"] == "Yes" and str(r["Pregnant"]).lower() == "yes":
                    reasons.append("Is pregnant.")

                if reasons:
                    return f"Volunteer {vid} is not eligible because: " + ", ".join(reasons)
                else:
                    return f"Volunteer {vid} is not eligible but specific reasons could not be determined."

        else:
            return "Volunteer ID not recognized."

    return None

# -------------------------- GPT QA FUNCTION --------------------------
def gpt_answer(q, df):
    llm = ChatOpenAI(temperature=0)
    volunteer_data = df[["VolunteerID", "Age", "Gender", "DiseaseStage", "BiomarkerStatus", "Region"]].to_string(index=False)

    prompt = PromptTemplate(
        template="""
        You are an AI assistant helping to answer queries about clinical trial volunteers.
        Data:
        {data}
        Question:
        {question}
        Answer:""",
        input_variables=["data", "question"]
    )

    final_prompt = prompt.format(data=volunteer_data, question=q)
    return llm.predict(final_prompt)

# -------------------------- ENTITY EXTRACTION --------------------------
def extract_entities(q):
    genders = []
    stages = []
    regions = []
    biomarkers = []
    ages = []

    q_lower = q.lower()

    if "male" in q_lower:
        genders.append("Male")
    if "female" in q_lower:
        genders.append("Female")

    for s in ["I", "II", "III", "IV"]:
        if f"stage {s}".lower() in q_lower:
            stages.append(s)

    for b in ["EGFR+", "ALK+", "KRAS+", "ROS1+"]:
        if b.lower() in q_lower:
            biomarkers.append(b)

    region_keywords = ["Delhi", "Mumbai", "Kolkata", "Chennai", "Hyderabad"]
    for r in region_keywords:
        if r.lower() in q_lower:
            regions.append(r)

    age_numbers = re.findall(r'(?:above|over|greater than|older than|age)\s*(\d+)', q_lower)
    if age_numbers:
        ages.append(int(age_numbers[0]))

    return {
        "gender": genders,
        "stage": stages,
        "region": regions,
        "biomarker": biomarkers,
        "age": ages
    }

# -------------------------- UI --------------------------
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# -------------------------- ADMIN PANEL --------------------------
with tab1:
    st.header("TechVitals Admin Portal")

    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)

    df = st.session_state.volunteers_df

    if df.empty:
        st.warning("Please upload a Volunteers CSV to continue.")
    else:
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

        st.subheader("Ask a Question (AI Powered)")
        q = st.text_input("Ask about volunteers:", key="admin_q")
        if st.button("Ask", key="admin_ask"):
            ans = answer_question(q, filtered)
            if ans is None:
                ans = gpt_answer(q, filtered)
            st.session_state.history_admin.append((q, ans))
            st.success(ans)

        if st.session_state.history_admin:
            with st.expander("Conversation History"):
                for ques, ans in st.session_state.history_admin:
                    st.markdown(f"**Q:** {ques}\n\n**A:** {ans}")

# -------------------------- UPDATED MEDICAL COMPANY PANEL --------------------------
with tab2:
    st.header("Medical Company Portal")

    if st.session_state.volunteers_df.empty:
        st.warning("Waiting for TechVitals to add the dataset...")
    else:
        st.write(f"TechVitals has provided a dataset with columns like {list(st.session_state.volunteers_df.columns)} and with {len(st.session_state.volunteers_df)} volunteers.")
        
        pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
        if pdf_file:
            try:
                with st.spinner("Processing PDF..."):
                    criteria, vectorstore = parse_pdf(pdf_file)
                    st.session_state.criteria = criteria
                    st.session_state.vectorstore = vectorstore
                    
                    # Display extracted criteria
                    with st.expander("Extracted Medicine Criteria", expanded=True):
                        st.markdown(f"""
                        **Inclusion Criteria:**
                        - Age: {criteria['min_age']}-{criteria['max_age']} years
                        - Condition: {criteria['condition']}
                        - Biomarker: {criteria['biomarker']}
                        - Stages: {', '.join(criteria['stages'])}
                        
                        **Exclusion Criteria:**
                        - Exclude Pregnant: {criteria['exclude_pregnant']}
                        - Exclude Diabetes: {criteria['exclude_diabetes']}
                        """)
                    st.success("Medical criteria processed successfully!")
                    
            except Exception as e:
                st.error(f"Invalid PDF: {str(e)}. Please upload a valid clinical trial criteria PDF.")
                st.session_state.criteria = None
                st.session_state.vectorstore = None

        if st.session_state.criteria:
            eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)
            st.success(f"Total eligible volunteers: {len(eligible.drop_duplicates('VolunteerID'))}")

            # Filters
            min_age_m, max_age_m = st.slider("Age Range", 0, 100,
                                             (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]),
                                             key="medical_age")

            gender_m = st.selectbox("Gender", ["All"] + sorted(eligible["Gender"].dropna().unique()), key="medical_gender")
            stage_m = st.selectbox("Disease Stage", ["All"] + sorted(eligible["DiseaseStage"].dropna().unique()),
                                   key="medical_stage")
            biomarker_m = st.selectbox("Biomarker", ["All"] + sorted(eligible["BiomarkerStatus"].dropna().unique()),
                                       key="medical_biomarker")
            region_m = st.selectbox("Region", ["All"] + sorted(eligible["Region"].dropna().unique()),
                                    key="medical_region")

            filtered_med = eligible[(eligible["Age"] >= min_age_m) & (eligible["Age"] <= max_age_m)]
            if gender_m != "All":
                filtered_med = filtered_med[filtered_med["Gender"] == gender_m]
            if stage_m != "All":
                filtered_med = filtered_med[filtered_med["DiseaseStage"] == stage_m]
            if biomarker_m != "All":
                filtered_med = filtered_med[filtered_med["BiomarkerStatus"] == biomarker_m]
            if region_m != "All":
                filtered_med = filtered_med[filtered_med["Region"] == region_m]

            st.subheader("Eligible Volunteers")
            st.dataframe(
                filtered_med[["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender", "Region", "Age"]]
                .drop_duplicates("VolunteerID")
            )

            st.subheader("Ask Trial Questions")
            q2 = st.text_input("Ask about eligible volunteers:", key="medical_q")
            if st.button("Ask", key="medical_ask"):
                ans2 = answer_question(q2, filtered_med, st.session_state.criteria)
                st.session_state.history_medical.append((q2, ans2))
                st.success(ans2)

            if st.session_state.history_medical:
                with st.expander("Conversation History"):
                    for ques2, ans2 in st.session_state.history_medical:
                        st.markdown(f"**Q:** {ques2}\n\n**A:** {ans2}")
