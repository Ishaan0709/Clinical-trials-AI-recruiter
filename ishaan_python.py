
import streamlit as st
import pandas as pd
import fitz
import re
import spacy
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="AI Clinical Trial Management System", layout="wide")
st.title("AI Clinical Trial Management System")

# -------------------------- SESSION STATE INIT --------------------------
if 'volunteers_df' not in st.session_state:
    try:
        st.session_state.volunteers_df = pd.read_csv("Realistic_Indian_Volunteers_BIG.csv")
    except:
        st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.history_admin = []
    st.session_state.history_medical = []
    st.session_state.criteria = None
    st.session_state.vectorstore = None

# -------------------------- LOAD SPACY --------------------------
nlp = spacy.load("en_core_web_sm")

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

# -------------------------- RULE-BASED QA FUNCTION --------------------------
def answer_question(q, df, criteria=None):
    q_lower = q.lower()

    if "how many" in q_lower:
        if "male" in q_lower:
            return f"There are {len(df[df['Gender'].str.lower() == 'male'])} eligible male volunteers."
        if "female" in q_lower:
            return f"There are {len(df[df['Gender'].str.lower() == 'female'])} eligible female volunteers."
        if "stage iii" in q_lower:
            return f"There are {len(df[df['DiseaseStage'] == 'III'])} eligible volunteers."
        age_numbers = re.findall(r'(?:above|over|greater than|older than|age)\s*(\d+)', q_lower)
        if age_numbers:
            age = int(age_numbers[0])
            return f"There are {len(df[df['Age'] > age])} eligible volunteers."
        return f"There are {len(df)} eligible volunteers."

    elif "list" in q_lower or "show" in q_lower:
        return ", ".join(df["VolunteerID"]) if not df.empty else "No eligible volunteers."

    elif "why is volunteer" in q_lower and "not eligible" in q_lower:
        vid = re.findall(r'volunteer (\w+)', q_lower)
        if vid:
            vid = vid[0]
            row = st.session_state.volunteers_df[st.session_state.volunteers_df["VolunteerID"] == vid]
            if row.empty:
                return f"Volunteer {vid} not found."
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
                return ", ".join(reasons) if reasons else "Volunteer meets all criteria."
        else:
            return "Volunteer ID not recognized."

    else:
        return None  # So that GPT can answer if rule-based fails

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



# -------------------------- NER FUNCTION --------------------------
def extract_entities(q):
    doc = nlp(q)
    genders = []
    stages = []
    regions = []
    biomarkers = []
    ages = []

    for ent in doc.ents:
        if ent.label_ == "CARDINAL" and ent.text.isdigit():
            ages.append(int(ent.text))
        if ent.label_ == "GPE":
            regions.append(ent.text)

    if "male" in q.lower():
        genders.append("Male")
    if "female" in q.lower():
        genders.append("Female")

    for s in ["I", "II", "III", "IV"]:
        if f"stage {s}" in q.lower():
            stages.append(s)

    for b in ["EGFR+", "ALK+", "KRAS+", "ROS1+"]:
        if b.lower() in q.lower():
            biomarkers.append(b)

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

# -------------------------- MEDICAL COMPANY PANEL --------------------------
with tab2:
    st.header("Medical Company Portal")

    pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=["pdf"])
    if pdf_file:
        with st.spinner("Processing PDF..."):
            criteria, vectorstore = parse_pdf(pdf_file)
            st.session_state.criteria = criteria
            st.session_state.vectorstore = vectorstore

    if st.session_state.criteria:
        st.subheader("Extracted Criteria")
        st.json(st.session_state.criteria)

        eligible = filter_volunteers(st.session_state.volunteers_df, st.session_state.criteria)

        # Medical panel filters
        st.subheader("Filter Eligible Volunteers")
        min_age_m, max_age_m = st.slider(
            "Age Range", 0, 100, 
            (st.session_state.criteria["min_age"], st.session_state.criteria["max_age"]),
            key="medical_age"
        )

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

        st.subheader("Eligible Volunteers (Privacy Protected)")
        st.dataframe(
            filtered_med[["VolunteerID", "Email", "DiseaseStage", "BiomarkerStatus", "Gender", "Region", "Age"]]
        )

        st.subheader("Ask a Question (AI Powered)")
        q2 = st.text_input("Ask about eligible volunteers:", key="medical_q")
        if st.button("Ask", key="medical_ask"):

            # Rule-based first
            ans2 = answer_question(q2, filtered_med, st.session_state.criteria)

            # If rule-based can't answer
            if ans2 is None:

                # Try NER
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
                    # Fallback to GPT if NER filter also fails
                    ans2 = gpt_answer(q2, filtered_med)

            st.session_state.history_medical.append((q2, ans2))
            st.success(ans2)

        if st.session_state.history_medical:
            with st.expander("Conversation History"):
                for ques2, ans2 in st.session_state.history_medical:
                    st.markdown(f"**Q:** {ques2}\n\n**A:** {ans2}")





