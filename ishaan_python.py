import streamlit as st
import pandas as pd
import fitz
import re

# Load default volunteer dataset
if 'volunteers_df' not in st.session_state:
    try:
        st.session_state.volunteers_df = pd.read_csv("Realistic_Indian_Volunteers.csv")
    except:
        st.session_state.volunteers_df = pd.DataFrame()
    st.session_state.dataset_qa_history = []
    st.session_state.history_matched = []
    st.session_state.matched_df = None
    st.session_state.inclusion_points = []
    st.session_state.exclusion_points = []
    st.session_state.mode = "Upload PDF"
    st.session_state.active_criteria = False
    st.session_state.last_pdf_name = None

# [Keep all the existing functions (parse_trial_pdf, filter_volunteers, answer_question_about_df) unchanged]

# UI Layout
st.set_page_config(layout="wide")
st.title("AI Clinical Trial Volunteer Recruiter")
tab1, tab2 = st.tabs(["📊 TechVitals Admin", "🏥 Medical Company"])

with tab1:
    st.header("TechVitals Admin Panel")
    st.write("Upload volunteer data or explore the existing dataset. Filter volunteers and ask questions using natural language.")
    
    with st.container(border=True):
        file_csv = st.file_uploader("Upload Volunteer CSV", type=['csv'], key="csv_uploader")
        if file_csv is not None:
            st.session_state.volunteers_df = pd.read_csv(file_csv)
    
    df_vol = st.session_state.volunteers_df
    if df_vol.empty:
        st.info("No volunteer data available.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            min_age_val = int(df_vol['Age'].min()) if not df_vol.empty else 0
            max_age_val = int(df_vol['Age'].max()) if not df_vol.empty else 100
            age_range = st.slider("Age Range", min_value=min_age_val, max_value=max_age_val, 
                                value=(min_age_val, max_age_val), key="admin_age")
        with col2:
            genders = ["All"] + sorted(df_vol['Gender'].dropna().unique().tolist())
            gender_choice = st.selectbox("Gender", options=genders, index=0, key="admin_gender")
        with col3:
            regions = ["All"] + sorted(df_vol['Region'].dropna().unique().tolist())
            region_choice = st.selectbox("Region", options=regions, index=0, key="admin_region")
        
        filtered_df = df_vol.copy()
        filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        if gender_choice != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == gender_choice]
        if region_choice != "All":
            filtered_df = filtered_df[filtered_df['Region'] == region_choice]
        
        with st.expander("📋 Volunteer List", expanded=True):
            st.dataframe(filtered_df.reset_index(drop=True), use_container_width=True)
        
        with st.container(border=True):
            st.subheader("💬 Dataset Q&A")
            with st.form(key="dataset_qa_form"):
                q1 = st.text_area("Ask about volunteers:", placeholder="e.g., How many female volunteers from Delhi have Stage III NSCLC?",
                                height=100, key="admin_question")
                submit_q1 = st.form_submit_button("Ask", use_container_width=True)
            
            if submit_q1 and q1:
                answer1 = answer_question_about_df(q1, df_vol)
                st.session_state.dataset_qa_history.append((q1, answer1))
                
            if st.session_state.dataset_qa_history:
                st.subheader("🗒️ Conversation History")
                for i, (ques, ans) in enumerate(reversed(st.session_state.dataset_qa_history)):
                    with st.expander(f"Q{i+1}: {ques[:50]}..."):
                        st.markdown(f"**Question:** {ques}")
                        st.markdown(f"**Answer:** {ans}")
                        st.divider()

with tab2:
    st.header("Medical Company Portal")
    st.write("Upload trial criteria or manually enter parameters to find eligible volunteers.")
    
    with st.container(border=True):
        mode = st.radio("Input Method:", ["Upload PDF", "Fill Form"], index=0, 
                        horizontal=True, key="mode_selector")
        if mode != st.session_state.mode:
            st.session_state.active_criteria = False
            st.session_state.matched_df = None
            st.session_state.inclusion_points = []
            st.session_state.exclusion_points = []
        st.session_state.mode = mode
        
        criteria = {}
        if mode == "Upload PDF":
            pdf_file = st.file_uploader("Upload Trial PDF", type=['pdf'], key="pdf_uploader")
            if pdf_file is not None:
                if st.session_state.last_pdf_name != pdf_file.name:
                    pdf_bytes = pdf_file.getvalue()
                    criteria, inc_points, exc_points = parse_trial_pdf(pdf_bytes)
                    st.session_state.inclusion_points = inc_points
                    st.session_state.exclusion_points = exc_points
                    st.session_state.matched_df = filter_volunteers(st.session_state.volunteers_df, criteria)
                    st.session_state.active_criteria = True
                    st.session_state.last_pdf_name = pdf_file.name
        else:
            with st.form(key="criteria_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age_min = st.number_input("Minimum Age", min_value=0, max_value=120, value=18, key="form_min_age")
                    cond_input = st.text_input("Required Condition", placeholder="e.g., NSCLC", key="form_condition")
                    stages_allowed = st.multiselect("Allowed Stages", ['I','II','III','IV'], default=['I','II','III','IV'], key="form_stages")
                with col2:
                    age_max = st.number_input("Maximum Age", min_value=0, max_value=120, value=100, key="form_max_age")
                    biomarker_input = st.text_input("Biomarker Status", placeholder="e.g., EGFR+", key="form_biomarker")
                    gender_allowed = st.selectbox("Allowed Gender", ["Any", "Male", "Female"], index=0, key="form_gender")
                
                exclude_diab = st.checkbox("Exclude diabetics", key="form_diabetes")
                exclude_preg = st.checkbox("Exclude pregnant", key="form_pregnant")
                submit_form = st.form_submit_button("Find Volunteers", use_container_width=True)
                
                if submit_form:
                    criteria = {
                        'min_age': int(age_min), 'max_age': int(age_max),
                        'conditions': [cond_input] if cond_input else [],
                        'biomarker': biomarker_input if biomarker_input else None,
                        'stages': stages_allowed if stages_allowed else [],
                        'gender': gender_allowed, 
                        'exclude_diabetes': exclude_diab, 'exclude_pregnant': exclude_preg
                    }
                    criteria['conditions'] = [c.strip() for c in criteria['conditions'] if c.strip()]
                    criteria['stages'] = [s.strip().upper() for s in criteria['stages']]
                    st.session_state.inclusion_points = []
                    st.session_state.exclusion_points = []
                    st.session_state.matched_df = filter_volunteers(st.session_state.volunteers_df, criteria)
                    st.session_state.active_criteria = True

    if st.session_state.active_criteria:
        with st.expander("🔍 Trial Criteria Summary", expanded=True):
            if mode == "Upload PDF" and st.session_state.inclusion_points:
                st.subheader("Inclusion Criteria")
                for pt in st.session_state.inclusion_points:
                    st.markdown(f"✅ {pt}")
                
                st.subheader("Exclusion Criteria")
                for pt in st.session_state.exclusion_points:
                    st.markdown(f"❌ {pt}")
            else:
                inc_list = []
                exc_list = []
                if criteria.get('min_age') is not None or criteria.get('max_age') is not None:
                    age_min_val = criteria.get('min_age', 0)
                    age_max_val = criteria.get('max_age', 120)
                    inc_list.append(f"Age between {age_min_val} and {age_max_val} years")
                if criteria.get('conditions'):
                    inc_list.append("Condition: " + ", ".join(criteria['conditions']))
                if criteria.get('biomarker'):
                    inc_list.append(f"Biomarker required: {criteria['biomarker']}")
                if criteria.get('stages'):
                    stages_str = ", ".join(criteria['stages'])
                    inc_list.append(f"Allowed Stages: {stages_str}")
                else:
                    inc_list.append("Allowed Stages: Any")
                if criteria.get('gender') and criteria['gender'] != "Any":
                    inc_list.append(f"Gender: {criteria['gender']} only")
                else:
                    inc_list.append("Gender: Any")
                if criteria.get('exclude_diabetes'):
                    exc_list.append("Excluding volunteers with diabetes")
                if criteria.get('exclude_pregnant'):
                    exc_list.append("Excluding pregnant volunteers")
                
                if inc_list:
                    st.markdown("**Inclusion Criteria:**")
                    for item in inc_list:
                        st.markdown(f"- {item}")
                if exc_list:
                    st.markdown("**Exclusion Criteria:**")
                    for item in exc_list:
                        st.markdown(f"- {item}")

        # Corrected eligible_df line
        eligible_df = st.session_state.matched_df.copy().reset_index(drop=True) if st.session_state.matched_df is not None else pd.DataFrame()
        
        with st.container(border=True):
            st.subheader("🧑⚕️ Eligible Volunteers")
            if eligible_df.empty:
                st.info("No matching volunteers found")
            else:
                col1, col2, col3, col4 = st.columns([2,1.5,1.5,1.5])
                with col1:
                    min_age2 = int(eligible_df['Age'].min())
                    max_age2 = int(eligible_df['Age'].max())
                    age_range2 = st.slider("Filter Age", min_age2, max_age2, (min_age2, max_age2), key="matched_age")
                with col2:
                    gender2 = ["All"] + sorted(eligible_df['Gender'].dropna().unique().tolist())
                    gender_choice2 = st.selectbox("Filter Gender", gender2, key="matched_gender")
                with col3:
                    regions2 = ["All"] + sorted(eligible_df['Region'].dropna().unique().tolist())
                    region_choice2 = st.selectbox("Filter Region", regions2, key="matched_region")
                with col4:
                    stage2 = ["All"] + sorted(eligible_df['DiseaseStage'].dropna().unique().tolist())
                    stage_choice2 = st.selectbox("Filter Stage", stage2, key="matched_stage")
                
                filtered_matched = eligible_df.copy()
                filtered_matched = filtered_matched[(filtered_matched['Age'] >= age_range2[0]) & (filtered_matched['Age'] <= age_range2[1])]
                if gender_choice2 != "All":
                    filtered_matched = filtered_matched[filtered_matched['Gender'] == gender_choice2]
                if region_choice2 != "All":
                    filtered_matched = filtered_matched[filtered_matched['Region'] == region_choice2]
                if stage_choice2 != "All":
                    filtered_matched = filtered_matched[filtered_matched['DiseaseStage'] == stage_choice2]
                
                st.write(f"🔢 Matching Volunteers: {len(filtered_matched)}")
                st.dataframe(filtered_matched, use_container_width=True)
                
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    csv_data = filtered_matched.to_csv(index=False)
                    st.download_button("📥 Download CSV", data=csv_data, file_name="Eligible_Volunteers.csv", 
                                      mime="text/csv", use_container_width=True)
                if mode == "Upload PDF" and pdf_file is not None:
                    with col_exp2:
                        st.download_button("📄 Download Criteria PDF", data=pdf_file.getvalue(), 
                                         file_name="Trial_Criteria.pdf", mime="application/pdf",
                                         use_container_width=True)

        with st.container(border=True):
            st.subheader("💬 Matched Volunteers Q&A")
            with st.form(key="matched_qa_form"):
                q2 = st.text_area("Ask about matches:", placeholder="e.g., List female volunteers from Mumbai with EGFR+",
                                height=100, key="matched_question")
                submit_q2 = st.form_submit_button("Ask", use_container_width=True)
            
            if submit_q2 and q2:
                full_matched_df = st.session_state.matched_df if st.session_state.matched_df is not None else filtered_matched
                answer2 = answer_question_about_df(q2, full_matched_df)
                st.session_state.history_matched.append((q2, answer2))
            
            if st.session_state.history_matched:
                st.subheader("🗒️ Conversation History")
                for i, (ques, ans) in enumerate(reversed(st.session_state.history_matched)):
                    with st.expander(f"Q{i+1}: {ques[:50]}...", expanded=False):
                        st.markdown(f"**Question:** {ques}")
                        st.markdown(f"**Answer:** {ans}")
                        st.divider()
