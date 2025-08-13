# 🧪 Clinical Trials AI Recruiter (MedMatch AI)

LLM-powered app to match **clinical trial criteria** (PDF) with **volunteer data** (CSV), filter results, and answer short questions.

## 🔗 Live
- Website (Main Webpage): https://ishaan0709.github.io/Techvital_clinical_trials/
- AI Chatbot (Streamlit): http://localhost:8501/  
  _Dev locally: http://localhost:8501

## ✨ What it does
- Upload Volunteers CSV → filter by age, gender, region
- Upload Trial Criteria PDF → parser extracts required fields
- Get eligible counts / shortlists and 2–3 sentence AI answers

## 🧰 Tech
Python, Streamlit, LangChain, langchain-openai, FAISS, pandas, PyMuPDF (fitz)

## 🚀 Run locally
```bash
python -m venv .venv
# Win: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run ishaan_python.py
