import streamlit as st
import pandas as pd
import os
import tempfile
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY or "sk-" not in OPENAI_API_KEY:
    raise ValueError("Invalid or missing OpenAI API key.")

embeddings = OpenAIEmbeddings()

# Helper function for matching volunteers

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
            if not gender_req or gender in gender_req:
                if (not exclude_diabetes or row["Diabetes"] == "No") and \
                   (not exclude_pregnant or row["Pregnant"] == "No"):
                    eligible.append(row)
    return pd.DataFrame(eligible)

# Load documents into FAISS

def load_documents(uploaded_files):
    rec_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_path = temp_file.name

        if uploaded_file.name.endswith(".pdf"):
            loader = PyMuPDFLoader(temp_path)
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents(docs)
            rec_chunks.extend(chunks)

        elif uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(temp_path)
            text = "\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1))
            doc = Document(page_content=text)
            splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
            chunks = splitter.split_documents([doc])
            rec_chunks.extend(chunks)

    return FAISS.from_documents(rec_chunks, embeddings)

def main():
    st.set_page_config(page_title="Clinical Trial AI Recruiter", layout="wide")
    st.title("🤖 AI Clinical Trial Volunteer Recruiter")

    tab1, tab2 = st.tabs(["👨‍💼 TechVitals Admin", "🏥 Medical Company"])

    with tab1:
        st.header("Upload Volunteer Dataset")
        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="admin_csv")
        if csv_file:
            df = pd.read_csv(csv_file)
            st.success("Dataset uploaded successfully!")
            st.dataframe(df)
            st.session_state.volunteer_df = df

            question = st.text_input("Ask a question to get filtered volunteers (e.g. age > 50):")
            if question:
                st.info("Full access granted (Admin view)")
                model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                documents = [Document(page_content="\n".join(df.astype(str).apply(lambda row: " | ".join(row), axis=1)))]
                vector = FAISS.from_documents(documents, embeddings)
                result = vector.similarity_search(question, k=5)
                answer = model.invoke(f"Based on the following data, answer the question with bullet points:\n{result[0].page_content}\nQuestion: {question}")
                st.markdown(answer.content)

    with tab2:
        st.header("Medical Company Trial Submission")
        mode = st.radio("Choose mode of input:", ["📄 Upload Trial PDF", "✍️ Fill Live Trial Form"])

        if mode == "📄 Upload Trial PDF":
            uploaded_pdfs = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)
            if uploaded_pdfs:
                index = load_documents(uploaded_pdfs)
                query = st.text_input("Ask about volunteers that match this trial:")
                if query and "volunteer_df" in st.session_state:
                    docs = index.similarity_search(query, k=5)
                    result = "\n\n".join([doc.page_content for doc in docs])
                    model = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
                    answer = model.invoke(f"Using the following trial criteria, list volunteer IDs and emails only that match:\n{result}\n\nVolunteers:\n{st.session_state.volunteer_df.to_csv(index=False)}")
                    st.markdown(answer.content)

        elif mode == "✍️ Fill Live Trial Form":
            with st.form("live_trial_form"):
                trial_name = st.text_input("Trial Name")
                drug = st.text_input("Drug Name")
                condition = st.text_input("Target Condition")
                phase = st.selectbox("Trial Phase", ["Phase 1", "Phase 2", "Phase 3", "Phase 4"])
                age_range = st.slider("Eligible Age Range", 0, 100, (18, 65))
                gender = st.multiselect("Eligible Genders", ["Male", "Female", "Other"])
                exclude_diabetes = st.checkbox("Exclude Diabetic Volunteers")
                exclude_pregnant = st.checkbox("Exclude Pregnant Volunteers")
                submitted = st.form_submit_button("Submit Trial")

            if submitted and "volunteer_df" in st.session_state:
                df = st.session_state.volunteer_df
                matches = match_volunteers(df, {
                    "age_range": age_range,
                    "condition": condition,
                    "gender": gender,
                    "exclude_diabetes": exclude_diabetes,
                    "exclude_pregnant": exclude_pregnant
                })
                st.success(f"{len(matches)} Volunteers matched this trial!")
                st.dataframe(matches[["VolunteerID", "Email"]])

if __name__ == "__main__":
    main()
