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
    st.session_state.pdf_summary = ""

# -------------------------- PDF CRITERIA EXTRACTION --------------------------

def extract_criteria_from_text(text):
    criteria = {
        "min_age": 0,
        "max_age": 100,
        "condition": "Any",
        "biomarker": "Any",
        "stages": [],
        "gender": "Any",
        "exclude_diabetes": "No",
        "exclude_pregnant": "No"
    }

    age_match = re.search(r'Age\s*[:\-]*\s*(\d+)\s*[\-to]*\s*(\d+)', text, re.IGNORECASE)
    if age_match:
        criteria["min_age"] = int(age_match.group(1))
        criteria["max_age"] = int(age_match.group(2))

    condition_match = re.search(r'Condition\s*[:\-]*\s*(.*)', text, re.IGNORECASE)
    if condition_match:
        criteria["condition"] = condition_match.group(1).split("\n")[0]

    biomarker_match = re.search(r'Biomarker\s*[:\-]*\s*([\w\+\-]+)', text, re.IGNORECASE)
    if biomarker_match:
        criteria["biomarker"] = biomarker_match.group(1)

    stages_match = re.findall(r'Stage\s*(I{1,3}|IV)', text, re.IGNORECASE)
    if stages_match:
        criteria["stages"] = [s.upper() for s in stages_match]

    if re.search(r'Gender\s*[:\-]*\s*Male', text, re.IGNORECASE):
        criteria["gender"] = "Male"
    elif re.search(r'Gender\s*[:\-]*\s*Female', text, re.IGNORECASE):
        criteria["gender"] = "Female"

    if re.search(r'Exclude\s*.*diabetes', text, re.IGNORECASE):
        criteria["exclude_diabetes"] = "Yes"
    if re.search(r'Exclude\s*.*pregnant', text, re.IGNORECASE):
        criteria["exclude_pregnant"] = "Yes"

    return criteria

def parse_pdf(uploaded_file):
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        text = ""
        for page in doc:
            text += page.get_text()

    criteria = extract_criteria_from_text(text)

    summary = f"""
    **Medical Trial Criteria Summary**
    - Age Range: {criteria['min_age']} to {criteria['max_age']}
    - Condition: {criteria['condition']}
    - Biomarker: {criteria['biomarker']}
    - Stages: {', '.join(criteria['stages']) if criteria['stages'] else 'Any'}
    - Gender: {criteria['gender']}
    - Exclude Diabetes: {criteria['exclude_diabetes']}
    - Exclude Pregnant: {criteria['exclude_pregnant']}
    """
    st.session_state.pdf_summary = summary

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.create_documents([text])
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    return criteria, vectorstore

# -------------------------- FILTER FUNCTION --------------------------

def filter_volunteers(df, criteria):
    result = df.copy()
    result = result[(result['Age'] >= criteria["min_age"]) & (result['Age'] <= criteria["max_age"])]
    if criteria["condition"] != "Any":
        result = result[result["Condition"].str.contains(criteria["condition"], case=False, na=False)]
    if criteria["biomarker"] != "Any":
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

# -------------------------- RULE-BASED QA FUNCTION --------------------------

def answer_question(q, df, criteria=None):
    q_lower = q.lower()

    if criteria:
        if any(word in q_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm your medical trial assistant."

        if "list" in q_lower or "show" in q_lower:
            clean_df = df.drop_duplicates("VolunteerID")
            if len(clean_df) > 15:
                return f"Found {len(clean_df)} eligible volunteers. Try refining your query."
            return ", ".join(clean_df["VolunteerID"].tolist())

        llm = ChatOpenAI(temperature=0)
        prompt = f"""
        MEDICAL CRITERIA:
        {criteria}

        VOLUNTEER DATA:
        {df[["VolunteerID", "Age", "Gender", "Condition", "BiomarkerStatus"]].to_string()}

        QUESTION: {q}

        ANSWER RULES:
        1. Reference criteria.
        2. Keep answers short.
        3. Use exact numbers.
        """
        return llm.predict(prompt)

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
                return f"Volunteer {vid} is fully eligible for the clinical trial."

            else:
                reasons = []
                r = row.iloc[0]
                if r["Age"] < criteria["min_age"] or r["Age"] > criteria["max_age"]:
                    reasons.append("Age not in eligible range.")
                if criteria["condition"].lower() not in str(r["Condition"]).lower():
                    reasons.append("Condition does not match.")
                if str(r["BiomarkerStatus"]).upper() != criteria["biomarker"].upper():
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
                    return f"Volunteer {vid} is not eligible but no specific reasons could be determined."
        else:
            return "Volunteer ID not recognized."

    return None

# -------------------------- GPT QA FUNCTION --------------------------

def gpt_answer(q, df):
    llm = ChatOpenAI(temperature=0)
    volunteer_data = df[["VolunteerID", "Age", "Gender", "DiseaseStage", "BiomarkerStatus", "Region"]].to_string(index=False)
    prompt = PromptTemplate(
        template="""
        You are an AI assistant answering clinical trial volunteer queries.
        Data:
        {data}
        Question:
        {question}
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

# -------------------------- MEDICAL COMPANY PANEL --------------------------

with tab2:
    st.header("Medical Company Portal")

    if st.session_state.volunteers_df.empty:
        st.warning("Waiting for TechVitals to add the dataset...")
    else:
        st.write(f"TechVitals has provided a dataset with {len(st.session_state.volunteers_df)} volunteers.")

        pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
        if pdf_file:
            with st.spinner("Processing PDF..."):
                criteria, vectorstore = parse_pdf(pdf_file)
                st.session_state.criteria = criteria
                st.session_state.vectorstore = vectorstore
                st.success("Medical criteria extracted successfully!")

        if st.session_state.pdf_summary:
            st.info(st.session_state.pdf_summary)

        if st.session_state.criteria:
            eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)
            st.success(f"Total eligible volunteers: {len(eligible.drop_duplicates('VolunteerID'))}")

            min_age_m, max_age_m = st.slider("Age Range", 0, 100,
                                             (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]),
                                             key="medical_age")
            gender_m = st.selectbox("Gender", ["All"] + sorted(eligible["Gender"].dropna().unique()), key="medical_gender")
            stage_m = st.selectbox("Disease Stage", ["All"] + sorted(eligible["DiseaseStage"].dropna().unique()), key="medical_stage")
            biomarker_m = st.selectbox("Biomarker", ["All"] + sorted(eligible["BiomarkerStatus"].dropna().unique()), key="medical_biomarker")
            region_m = st.selectbox("Region", ["All"] + sorted(eligible["Region"].dropna().unique()), key="medical_region")

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

            st.subheader("Ask Trial Questions (AI Powered)")
            q2 = st.text_input("Ask about eligible volunteers:", key="medical_q")
            if st.button("Ask", key="medical_ask"):
                ans2 = answer_question(q2, filtered_med, st.session_state.criteria)
                if ans2 is None:
                    ans2 = gpt_answer(q2, filtered_med)
                st.session_state.history_medical.append((q2, ans2))
                st.success(ans2)

            if st.session_state.history_medical:
                with st.expander("Conversation History"):
                    for ques2, ans2 in st.session_state.history_medical:
                        st.markdown(f"**Q:** {ques2}\n\n**A:** {ans2}")
