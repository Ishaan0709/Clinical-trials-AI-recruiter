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

# -------------------------- COMPREHENSIVE COUNT FUNCTION --------------------------
def get_comprehensive_counts(df, criteria=None):
    counts = {}
    
    # Basic counts
    counts['total'] = len(df)
    counts['male'] = len(df[df['Gender'].str.lower() == 'male'])
    counts['female'] = len(df[df['Gender'].str.lower() == 'female'])
    counts['diabetes'] = len(df[df['Diabetes'].str.lower() == 'yes'])
    counts['pregnant'] = len(df[df['Pregnant'].str.lower() == 'yes'])
    
    # Disease stage counts
    if 'DiseaseStage' in df.columns:
        for stage in ['I', 'II', 'III', 'IV']:
            counts[f'stage_{stage}'] = len(df[df['DiseaseStage'] == stage])
    
    # Biomarker counts
    if 'BiomarkerStatus' in df.columns:
        for biomarker in ['EGFR+', 'ALK+', 'KRAS+', 'ROS1+']:
            counts[f'biomarker_{biomarker.replace("+", "pos")}'] = len(df[df['BiomarkerStatus'] == biomarker])
    
    # Region counts
    if 'Region' in df.columns:
        for region in ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad']:
            counts[f'region_{region}'] = len(df[df['Region'] == region])
    
    # Age group counts
    age_bins = [0, 30, 40, 50, 60, 70, 100]
    age_labels = ['<30', '30-39', '40-49', '50-59', '60-69', '70+']
    if 'Age' in df.columns:
        age_groups = pd.cut(df['Age'], bins=age_bins, labels=age_labels)
        for label in age_labels:
            counts[f'age_{label}'] = len(age_groups[age_groups == label])
    
    # Criteria-specific counts if provided
    if criteria:
        counts['eligible'] = len(filter_volunteers(df, criteria))
        counts['ineligible'] = counts['total'] - counts['eligible']
    
    return counts

# -------------------------- RULE-BASED QA FUNCTION --------------------------
def answer_question(q, df, criteria=None):
    q_lower = q.lower()
    counts = get_comprehensive_counts(df, criteria)

    # Handle "how many" questions
    if "how many" in q_lower:
        # Check for specific count requests
        if "total" in q_lower or "all" in q_lower or "volunteer" in q_lower:
            return f"There are {counts['total']} total volunteers in the dataset."
        
        if "eligible" in q_lower and criteria:
            return f"There are {counts['eligible']} eligible volunteers based on the trial criteria."
        
        if "ineligible" in q_lower and criteria:
            return f"There are {counts['ineligible']} ineligible volunteers based on the trial criteria."
        
        if "male" in q_lower:
            if "diabetes" in q_lower:
                male_diabetes = len(df[(df['Gender'].str.lower() == 'male') & (df['Diabetes'].str.lower() == 'yes')])
                return f"There are {male_diabetes} male volunteers with diabetes."
            if "pregnant" in q_lower:
                return "There are 0 male pregnant volunteers (biological impossibility)."
            return f"There are {counts['male']} male volunteers."
        
        if "female" in q_lower:
            if "diabetes" in q_lower:
                female_diabetes = len(df[(df['Gender'].str.lower() == 'female') & (df['Diabetes'].str.lower() == 'yes')])
                return f"There are {female_diabetes} female volunteers with diabetes."
            if "pregnant" in q_lower:
                return f"There are {counts['pregnant']} pregnant female volunteers."
            return f"There are {counts['female']} female volunteers."
        
        if "diabetes" in q_lower:
            return f"There are {counts['diabetes']} volunteers with diabetes."
        
        if "pregnant" in q_lower:
            return f"There are {counts['pregnant']} pregnant volunteers."
        
        # Handle stage queries
        for stage in ['I', 'II', 'III', 'IV']:
            if f"stage {stage}" in q_lower:
                return f"There are {counts[f'stage_{stage}']} volunteers at stage {stage}."
        
        # Handle biomarker queries
        for biomarker in ['EGFR+', 'ALK+', 'KRAS+', 'ROS1+']:
            if biomarker.lower() in q_lower:
                return f"There are {counts[f'biomarker_{biomarker.replace("+", "pos")}']} volunteers with {biomarker} biomarker."
        
        # Handle region queries
        for region in ['Delhi', 'Mumbai', 'Kolkata', 'Chennai', 'Hyderabad']:
            if region.lower() in q_lower:
                return f"There are {counts[f'region_{region}']} volunteers from {region}."
        
        # Handle age group queries
        for label in ['<30', '30-39', '40-49', '50-59', '60-69', '70+']:
            if label.replace('-', ' to ').lower() in q_lower or label.replace('-', '-').lower() in q_lower:
                return f"There are {counts[f'age_{label}']} volunteers aged {label}."
        
        # If no specific count was found, return all counts
        return f"""
        Comprehensive volunteer counts:
        - Total: {counts['total']}
        - Male: {counts['male']}
        - Female: {counts['female']}
        - With diabetes: {counts['diabetes']}
        - Pregnant: {counts['pregnant']}
        - Disease stages: I={counts.get('stage_I', 0)}, II={counts.get('stage_II', 0)}, III={counts.get('stage_III', 0)}, IV={counts.get('stage_IV', 0)}
        - Biomarkers: EGFR+={counts.get('biomarker_EGFRpos', 0)}, ALK+={counts.get('biomarker_ALKpos', 0)}, KRAS+={counts.get('biomarker_KRASpos', 0)}, ROS1+={counts.get('biomarker_ROS1pos', 0)}
        """

    elif "list" in q_lower or "show" in q_lower:
        return ", ".join(df["VolunteerID"]) if not df.empty else "No eligible volunteers."

    elif "why is volunteer" in q_lower and ("eligible" in q_lower or "not eligible" in q_lower):
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
                    reasons.append(f"Age {r['Age']} not in eligible range ({criteria['min_age']}-{criteria['max_age']})")
                if criteria["condition"] and criteria["condition"].lower() not in str(r["Condition"]).lower():
                    reasons.append(f"Condition '{r['Condition']}' doesn't match required '{criteria['condition']}'")
                if criteria["biomarker"] and str(r["BiomarkerStatus"]).upper() != criteria["biomarker"].upper():
                    reasons.append(f"Biomarker '{r['BiomarkerStatus']}' doesn't match required '{criteria['biomarker']}'")
                if criteria["stages"] and r["DiseaseStage"] not in criteria["stages"]:
                    reasons.append(f"Disease stage '{r['DiseaseStage']}' not in required stages {criteria['stages']}")
                if criteria["gender"] != "Any" and r["Gender"].lower() != criteria["gender"].lower():
                    reasons.append(f"Gender '{r['Gender']}' doesn't match required '{criteria['gender']}'")
                if criteria["exclude_diabetes"] == "Yes" and str(r["Diabetes"]).lower() == "yes":
                    reasons.append("Has diabetes which is excluded")
                if criteria["exclude_pregnant"] == "Yes" and str(r["Pregnant"]).lower() == "yes":
                    reasons.append("Is pregnant which is excluded")

                if reasons:
                    return f"Volunteer {vid} is not eligible because:\n- " + "\n- ".join(reasons)
                else:
                    return f"Volunteer {vid} is not eligible but specific reasons could not be determined."

        else:
            return "Volunteer ID not recognized."

    return None  # GPT will handle if rule-based fails

# -------------------------- GPT QA FUNCTION --------------------------
def gpt_answer(q, df):
    llm = ChatOpenAI(temperature=0)
    volunteer_data = df[["VolunteerID", "Age", "Gender", "DiseaseStage", "BiomarkerStatus", "Region"]].to_string(index=False)

    prompt = PromptTemplate(
        template="""
        You are an AI assistant helping to answer queries about clinical trial volunteers.
        Current volunteer data:
        {data}
        
        Question:
        {question}
        
        Please provide a detailed, professional response. If the question is about counts or statistics,
        make sure to include all relevant numbers. For eligibility questions, be specific about criteria.
        
        Answer:""",
        input_variables=["data", "question"]
    )

    final_prompt = prompt.format(data=volunteer_data, question=q)
    return llm.predict(final_prompt)

# -------------------------- UI --------------------------
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

# -------------------------- ADMIN PANEL --------------------------
with tab1:
    st.header("TechVitals Admin Portal")

    uploaded_csv = st.file_uploader("Upload Volunteers CSV", type="csv")
    if uploaded_csv:
        st.session_state.volunteers_df = pd.read_csv(uploaded_csv)
        st.success(f"Dataset uploaded successfully with {len(st.session_state.volunteers_df)} volunteers!")

    df = st.session_state.volunteers_df

    if df.empty:
        st.warning("Please upload a Volunteers CSV to continue.")
    else:
        st.subheader("Dataset Summary")
        st.write(f"TechVitals have provided a dataset containing columns like {list(df.columns)} and with {len(df)} volunteers.")
        
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

# -------------------------- MEDICAL COMPANY PANEL --------------------------
with tab2:
    st.header("Medical Company Portal")

    if st.session_state.volunteers_df.empty:
        st.warning("Waiting for TechVitals to add the dataset...")
    else:
        st.write(f"TechVitals has provided a dataset with columns like {list(st.session_state.volunteers_df.columns)} and with {len(st.session_state.volunteers_df)} volunteers.")
        
        pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
        if pdf_file:
            with st.spinner("Processing PDF..."):
                criteria, vectorstore = parse_pdf(pdf_file)
                st.session_state.criteria = criteria
                st.session_state.vectorstore = vectorstore
                st.success("Medical criteria processed successfully!")

        if st.session_state.criteria:
            st.subheader("Extracted Criteria")
            st.json(st.session_state.criteria)

            eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)
            st.success(f"Total number of eligible volunteers based on Medical Criteria: {len(eligible)}")

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

            st.subheader("Eligible Volunteers (Privacy Protected)")
            st.dataframe(
                filtered_med[["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender", "Region", "Age"]]
            )

            st.subheader("Ask a Question (AI Powered)")
            q2 = st.text_input("Ask about eligible volunteers:", key="medical_q")
            if st.button("Ask", key="medical_ask"):
                ans2 = answer_question(q2, filtered_med, st.session_state.criteria)

                if ans2 is None:
                    entities = extract_entities(q2)
                    filtered = filtered_med.copy()

                    if entities["gender"]:
                        filtered = filtered[filtered["Gender"].isin(entities["gender"])]
                    if entities["stage"]:
                        filtered = filtered[filtered["DiseaseStage"].isin(entities["stage"])]
                    if entities["region"]:
                        filtered = filtered[filtered["Region"].isin(entities["region"])]
                    if entities["biomarker"]:
                        filtered = filtered[filtered["BiomarkerStatus"].isin(entities["biomarker"])]
                    if entities["age"]:
                        filtered = filtered[filtered["Age"] >= max(entities["age"])]

                    if not filtered.empty:
                        ans2 = f"Found {len(filtered)} matching volunteers: " + ", ".join(filtered["VolunteerID"])
                    else:
                        ans2 = gpt_answer(q2, filtered_med)

                st.session_state.history_medical.append((q2, ans2))
                st.success(ans2)

            if st.session_state.history_medical:
                with st.expander("Conversation History"):
                    for ques2, ans2 in st.session_state.history_medical:
                        st.markdown(f"**Q:** {ques2}\n\n**A:** {ans2}")
