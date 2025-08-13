# Clinical Trials AI Recruiter (MedMatch AI)

**Volunteer recruitment AI** for clinical trials. Uses **RAG (Retrieval-Augmented Generation)** on trial-criteria **PDFs** + volunteer **CSV** to return fast eligibility counts, shortlists, and 2–3 sentence answers.

## 🔗 Live
- Website (Landing): https://ishaan0709.github.io/Techvital_clinical_trials/
- AI App (Streamlit): http://localhost:8501
-  _Dev locally:_ http://localhost:8501

## ⚙️ Stack
Streamlit • Python • LangChain • **RAG** • FAISS • OpenAI • PyMuPDF (fitz) • pandas

## 🧠 How it works
1. Parse criteria **PDF** → extract: `Age between 30 and 75 years`, `Non-Small Cell Lung Cancer (NSCLC)`, `EGFR-positive (EGFR+)`, `Stage III or IV`.  
2. Chunk text → **FAISS** embeddings; query with **LangChain + OpenAI**.  
3. Filter CSV by age/gender/region/stage/biomarker; answer with concise summaries.

## 💬 Try in the Medical tab
- “How many eligible volunteers?”  
- “Eligible **EGFR+ Stage IV** volunteers?”  
- “Eligible females from **Mumbai** (show up to 5 IDs).”

