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

# Function: Parse trial criteria PDF
def parse_trial_pdf(pdf_bytes):
    # Extract text from PDF
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    # Split into inclusion and exclusion lines
    inc_points = []
    exc_points = []
    low_text = text.lower()
    inc_idx = low_text.find("inclusion criteria")
    exc_idx = low_text.find("exclusion criteria")
    if inc_idx != -1:
        # Start after "Inclusion Criteria:" heading
        start = text.find("\n", inc_idx) if text.find("\n", inc_idx) != -1 else inc_idx
        inc_section = text[start: exc_idx if exc_idx!=-1 else len(text)]
        for line in inc_section.splitlines():
            line = line.strip()
            if not line or line.lower().startswith("exclusion"):
                break
            if line.startswith("-"):
                inc_points.append(line.lstrip("- ").strip())
    if exc_idx != -1:
        start = text.find("\n", exc_idx) if text.find("\n", exc_idx) != -1 else exc_idx
        end_idx = low_text.find("endpoint", exc_idx)
        if end_idx == -1: end_idx = len(text)
        exc_section = text[start: end_idx]
        for line in exc_section.splitlines():
            line = line.strip()
            if not line: 
                continue
            if line.lower().startswith("endpoint"):
                break
            if line.startswith("-"):
                exc_points.append(line.lstrip("- ").strip())
    # Initialize criteria dict
    criteria = {
        'min_age': None, 'max_age': None,
        'conditions': [], 'biomarker': None, 'stages': [], 'gender': 'Any',
        'exclude_diabetes': False, 'exclude_pregnant': False
    }
    # Helper for stage parsing
    def parse_stage_line(line):
        stages = []
        if "Stage " in line:
            after = line.split("Stage ", 1)[1]
            parts = re.split(r'\bor\b|,', after)
            for part in parts:
                token = part.strip().strip('.')
                if not token:
                    continue
                token_up = token.upper()
                roman_set = ["I","II","III","IV"]
                roman_map = {"1":"I","2":"II","3":"III","4":"IV"}
                if token_up in roman_set:
                    stages.append(token_up)
                elif token_up.isdigit():
                    if token_up in roman_map:
                        stages.append(roman_map[token_up])
                else:
                    m = re.match(r'^(I{1,3}|IV)\b', token_up)
                    if m:
                        stages.append(m.group(1))
        # Return unique while preserving order
        return list(dict.fromkeys(stages))
    # Parse inclusion criteria points
    for inc in inc_points:
        low = inc.lower()
        if "age" in low:
            # Age range
            nums = re.findall(r'\d+', inc)
            if len(nums) >= 2:
                criteria['min_age'] = int(nums[0]); criteria['max_age'] = int(nums[1])
            elif len(nums) == 1:
                val = int(nums[0])
                if any(word in low for word in ["above", "over", "older than", "at least"]):
                    criteria['min_age'] = val
                elif any(word in low for word in ["below", "under", "younger than", "at most"]):
                    criteria['max_age'] = val
        elif "cancer" in low or "diagnosed with" in low or "indication" in low or "with " in low:
            # Condition/disease
            cond_part = inc
            if "with" in low:
                cond_part = inc.split("with", 1)[1].strip()
            cond_part = cond_part.strip().strip('.')
            # Remove any trailing stage info from condition part, if present
            cond_part = re.split(r'stage\s', cond_part, flags=re.IGNORECASE)[0].strip()
            if cond_part:
                if "(" in cond_part:
                    base = cond_part.split("(")[0].strip()
                    acronym = cond_part.split("(")[1].split(")")[0].strip()
                    criteria['conditions'].append(base)
                    criteria['conditions'].append(acronym)
                else:
                    criteria['conditions'].append(cond_part)
        elif any(marker in low for marker in ["egfr", "ALK".lower(), "HER2".lower(), "kras"]):
            # Biomarker status
            if "positive" in low or "+" in inc:
                if "egfr" in low:
                    criteria['biomarker'] = "EGFR+"
                elif "alk" in low:
                    criteria['biomarker'] = "ALK+"
                elif "her2" in low:
                    criteria['biomarker'] = "HER2+"
                elif "kras" in low:
                    criteria['biomarker'] = "KRAS+"
            elif "negative" in low or "-" in inc:
                if "egfr" in low:
                    criteria['biomarker'] = "EGFR-"
                elif "alk" in low:
                    criteria['biomarker'] = "ALK-"
                elif "her2" in low:
                    criteria['biomarker'] = "HER2-"
                elif "kras" in low:
                    criteria['biomarker'] = "KRAS-"
        elif "stage" in low:
            stages_found = parse_stage_line(inc)
            for s in stages_found:
                criteria['stages'].append(s)
        elif "gender" in low or "male" in low or "female" in low:
            if "any gender" in low or "male or female" in low or "both genders" in low:
                criteria['gender'] = "Any"
            elif "female" in low and "male" not in low:
                criteria['gender'] = "Female"
            elif "male" in low and "female" not in low:
                criteria['gender'] = "Male"
    # Parse exclusion criteria points
    for exc in exc_points:
        low = exc.lower()
        if "pregnant" in low or "breastfeeding" in low:
            criteria['exclude_pregnant'] = True
        if "diabetes" in low or "diabetic" in low:
            criteria['exclude_diabetes'] = True
    # Ensure unique entries in conditions and stages
    criteria['conditions'] = list(dict.fromkeys([c.strip() for c in criteria['conditions'] if c.strip()]))
    criteria['stages'] = list(dict.fromkeys([s.strip().upper() for s in criteria['stages'] if s.strip()]))
    # Fallback: if stage criteria not captured but inclusion line contains "Stage"
    if not criteria['stages']:
        for line in inc_points:
            if re.search(r'stage\s', line, flags=re.IGNORECASE):
                for s in parse_stage_line(line):
                    criteria['stages'].append(s)
        criteria['stages'] = list(dict.fromkeys(criteria['stages']))
    return criteria, inc_points, exc_points

# Function: Filter volunteers DataFrame based on criteria
def filter_volunteers(df, criteria):
    result = df.copy()
    if criteria.get('min_age') is not None:
        result = result[result['Age'] >= criteria['min_age']]
    if criteria.get('max_age') is not None:
        result = result[result['Age'] <= criteria['max_age']]
    if criteria.get('conditions'):
        cond_list = [c.lower() for c in criteria['conditions']]
        result = result[result['Condition'].str.lower().isin(cond_list)]
    if criteria.get('biomarker'):
        biomarker = criteria['biomarker']
        if biomarker.endswith('-'):
            marker = biomarker[:-1]
            # exclude those with the marker positive
            result = result[~result['BiomarkerStatus'].str.upper().str.contains(marker.upper() + '\+')]
        else:
            result = result[result['BiomarkerStatus'].str.upper() == biomarker.upper()]
    if criteria.get('stages'):
        stage_list = [s.upper() for s in criteria['stages']]
        result = result[result['DiseaseStage'].str.upper().isin(stage_list)]
    if criteria.get('gender') and criteria['gender'] != "Any":
        result = result[result['Gender'].str.lower() == criteria['gender'].lower()]
    if criteria.get('exclude_diabetes'):
        result = result[result['Diabetes'].str.lower() == 'no']
    if criteria.get('exclude_pregnant'):
        result = result[result['Pregnant'].str.lower() == 'no']
    return result

# Function: Q&A about a DataFrame
def answer_question_about_df(q, df):
    q = q.strip()
    if q.endswith('?'):
        q = q[:-1]
    q_lower = q.lower()
    # Personal question check (specific volunteer name)
    found_persons = []
    for idx, name in enumerate(df['Name']):
        name_lower = name.lower()
        if re.search(r'\b' + re.escape(name_lower) + r'\b', q_lower):
            found_persons = [(idx, name)]
            break
        first = name_lower.split()[0]
        if re.search(r'\b' + re.escape(first) + r'\b', q_lower):
            found_persons.append((idx, name))
    found_persons = list({p[0]: p for p in found_persons}.values())
    if len(found_persons) == 1:
        idx, name = found_persons[0]
        person = df.iloc[idx]
        gender = str(person['Gender'])
        if any(term in q_lower for term in ["age", "how old"]):
            return f"{name} is {person['Age']} years old."
        if "diab" in q_lower:
            return ("Yes, " + name + " has diabetes.") if str(person['Diabetes']).lower() == "yes" else ("No, " + name + " does not have diabetes.")
        if "pregnan" in q_lower:
            if gender.lower() == 'male':
                return f"{name} is male, so pregnancy is not applicable."
            else:
                return ("Yes, " + name + " is pregnant.") if str(person['Pregnant']).lower() == "yes" else ("No, " + name + " is not pregnant.")
        if "stage" in q_lower:
            return f"{name} is at Disease Stage {person['DiseaseStage']}."
        if "condition" in q_lower or "diagnosis" in q_lower:
            return f"{name} has been diagnosed with {person['Condition']}."
        if "biomarker" in q_lower or "mutation" in q_lower or "egfr" in q_lower or "alk" in q_lower or "her2" in q_lower or "kras" in q_lower:
            return f"{name}'s biomarker status is {person['BiomarkerStatus']}."
        if "where" in q_lower or "from" in q_lower:
            return f"{name} is from {person['Region']}."
        if "gender" in q_lower:
            return f"{name} is {person['Gender']}."
        # General info summary if no specific asked
        age = person['Age']; region = person['Region']; cond = person['Condition']; stage = person['DiseaseStage']; bio = person['BiomarkerStatus']; diab = str(person['Diabetes']).lower(); preg = str(person['Pregnant']).lower()
        pronoun = "He" if gender.lower()=="male" else "She"
        summary = f"{name} is a {age}-year-old {gender} from {region}, diagnosed with {cond} (Stage {stage}). Biomarker status is {bio}. "
        summary += (pronoun + " has diabetes. ") if diab == "yes" else (pronoun + " does not have diabetes. ")
        if gender.lower()=="female":
            summary += pronoun + (" is pregnant." if preg == "yes" else " is not pregnant.")
        return summary
    # General question (multiple volunteers)
    sub_df = df.copy()
    gender_filter = None; region_filter = None; condition_filter = None; stage_filter = []; biomarker_filter = None
    diabetes_filter = None; pregnant_filter = None
    age_min = None; age_max = None
    # Detect filters
    if re.search(r'\b(women|woman|female)\b', q_lower):
        gender_filter = "Female"
    if re.search(r'\b(men|man|male)\b', q_lower):
        if gender_filter == "Female":
            gender_filter = None  # both genders mentioned
        else:
            gender_filter = "Male"
    if re.search(r'\b(other|non-binary)\b', q_lower):
        if gender_filter:
            gender_filter = None
        else:
            gender_filter = "Other"
    # Region
    unique_regions = df['Region'].dropna().unique()
    found_region = None
    for reg in unique_regions:
        if re.search(r'\b' + re.escape(reg.lower()) + r'\b', q_lower):
            found_region = reg; break
    if found_region:
        region_filter = found_region
    else:
        m = re.search(r'\b(from|in)\s+([\w\s]+)', q_lower)
        if m:
            reg_candidate = m.group(2).strip()
            reg_candidate = re.sub(r'\bvolunteers?\b', '', reg_candidate).strip()
            if reg_candidate:
                region_filter = reg_candidate.title()
    # Condition
    unique_conditions = df['Condition'].dropna().unique()
    found_cond = None
    for cond in unique_conditions:
        if re.search(r'\b' + re.escape(cond.lower()) + r'\b', q_lower):
            found_cond = cond; break
    if "lung cancer" in q_lower:
        condition_filter = ["Lung Cancer", "NSCLC"]
    elif found_cond:
        condition_filter = [found_cond]
    elif re.search(r'\bcancer\b', q_lower) and not any(term in q_lower for term in ["lung", "breast", "colorectal", "leukemia", "nsclc"]):
        condition_filter = None  # 'cancer' generic (everyone has some cancer here)
    # Stage
    stage_values = set()
    roman_map = {"1":"I","2":"II","3":"III","4":"IV"}
    roman_matches = re.findall(r'\b(I{1,3}|IV)\b', q_lower, flags=re.I)
    for rm in roman_matches:
        stage_values.add(rm.upper())
    numeric_matches = re.findall(r'\b[1-4]\b', q_lower)
    for nm in numeric_matches:
        if nm in roman_map:
            stage_values.add(roman_map[nm])
    if stage_values:
        stage_filter = list(stage_values)
    # Biomarker
    for marker in ["EGFR","ALK","HER2","KRAS"]:
        if marker.lower() in q_lower:
            if "negative" in q_lower:
                biomarker_filter = marker + "-"
            else:
                biomarker_filter = marker + "+"
            break
    # Diabetes/Pregnant
    if "diab" in q_lower:
        if any(word in q_lower for word in ["no diab", "without diab", "no diabet", "not diabetic"]):
            diabetes_filter = False
        else:
            diabetes_filter = True
    if "pregnan" in q_lower:
        if any(word in q_lower for word in ["not pregnant", "no pregnant", "without pregnant"]):
            pregnant_filter = False
        else:
            pregnant_filter = True
    # Age
    m = re.search(r'between (\d+)\D+(\d+)', q_lower)
    if m:
        age_min = int(m.group(1)); age_max = int(m.group(2))
    m = re.search(r'(\d+)\s*-\s*(\d+)\s*year', q_lower)
    if m:
        age_min = int(m.group(1)); age_max = int(m.group(2))
    m = re.search(r'(above|over|older than)\s+(\d+)', q_lower)
    if m:
        age_min = int(m.group(2)) + 1
    m = re.search(r'(under|below|younger than)\s+(\d+)', q_lower)
    if m:
        age_max = int(m.group(2)) - 1
    m = re.search(r'at least\s+(\d+)', q_lower)
    if m:
        age_min = int(m.group(1))
    m = re.search(r'at most\s+(\d+)', q_lower)
    if m:
        age_max = int(m.group(1))
    m = re.search(r'\b(\d+)\s*(year old|years old)\b', q_lower)
    if m:
        age_min = age_max = int(m.group(1))
    # Apply filters to sub_df
    if age_min is not None: sub_df = sub_df[sub_df['Age'] >= age_min]
    if age_max is not None: sub_df = sub_df[sub_df['Age'] <= age_max]
    if gender_filter: sub_df = sub_df[sub_df['Gender'].str.lower() == gender_filter.lower()]
    if region_filter: sub_df = sub_df[sub_df['Region'].str.lower() == region_filter.lower()]
    if condition_filter:
        cond_list = [c.lower() for c in condition_filter]
        sub_df = sub_df[sub_df['Condition'].str.lower().isin(cond_list)]
    if stage_filter:
        sub_df = sub_df[sub_df['DiseaseStage'].str.upper().isin([s.upper() for s in stage_filter])]
    if biomarker_filter:
        if biomarker_filter.endswith('-'):
            marker = biomarker_filter[:-1]
            sub_df = sub_df[~sub_df['BiomarkerStatus'].str.upper().str.contains(marker + '\+')]
        else:
            sub_df = sub_df[sub_df['BiomarkerStatus'].str.upper() == biomarker_filter.upper()]
    if diabetes_filter is not None:
        if diabetes_filter: sub_df = sub_df[sub_df['Diabetes'].str.lower() == 'yes']
        else: sub_df = sub_df[sub_df['Diabetes'].str.lower() == 'no']
    if pregnant_filter is not None:
        if pregnant_filter: sub_df = sub_df[sub_df['Pregnant'].str.lower() == 'yes']
        else: sub_df = sub_df[sub_df['Pregnant'].str.lower() == 'no']
    total = len(sub_df)
    # Percentage questions
    if "percent" in q_lower or "%" in q_lower:
        match = re.search(r'what percentage of (.+) are (.+)', q_lower)
        if not match:
            match = re.search(r'what percentage of (.+) have (.+)', q_lower)
        if match:
            groupX = match.group(1).strip()
            groupY = match.group(2).strip()
            denom_df = df.copy()
            # Determine denominator filters from groupX text
            denom_filters = {}
            if re.search(r'\bfemale\b', groupX): denom_filters['Gender'] = "Female"
            if re.search(r'\bmale\b', groupX): denom_filters['Gender'] = "Male"
            if re.search(r'\bother\b', groupX): denom_filters['Gender'] = "Other"
            for reg in unique_regions:
                if reg.lower() in groupX:
                    denom_filters['Region'] = reg
            for cond in unique_conditions:
                if cond.lower() in groupX:
                    denom_filters['Condition'] = cond
            if "lung cancer" in groupX:
                denom_filters['Condition'] = "Lung Cancer"
            if "nsclc" in groupX:
                denom_filters['Condition'] = "NSCLC"
            if "diabet" in groupX: denom_filters['Diabetes'] = True
            if "pregnan" in groupX: denom_filters['Pregnant'] = True
            # Apply denom filters to denom_df
            for key, val in denom_filters.items():
                if key == 'Gender':
                    denom_df = denom_df[denom_df['Gender'].str.lower() == str(val).lower()]
                elif key == 'Region':
                    denom_df = denom_df[denom_df['Region'].str.lower() == str(val).lower()]
                elif key == 'Condition':
                    denom_df = denom_df[denom_df['Condition'].str.lower() == str(val).lower()]
                elif key == 'Diabetes':
                    denom_df = denom_df[denom_df['Diabetes'].str.lower() == 'yes']
                elif key == 'Pregnant':
                    denom_df = denom_df[denom_df['Pregnant'].str.lower() == 'yes']
            # Numerator is current sub_df (with all filters from question)
            num_df = sub_df
            if len(denom_df) == 0:
                return f"No volunteers in the category: {groupX}."
            percent = (len(num_df) / len(denom_df)) * 100
            # Format groupX and groupY for answer
            outX = groupX if groupX else "volunteers"
            outY = groupY
            # Capitalize region names in outX/outY if present
            if " from " in outX:
                parts = outX.split(" from ")
                outX = parts[0].strip() + " from " + parts[1].strip().title()
            if outX == "" or outX.lower().startswith("volunteer"):
                outX = "volunteers"
            if outX in ["female", "male", "other"]:
                outX += " volunteers"
            if outY.lower().startswith("from "):
                outY = "from " + outY[5:].strip().title()
            answer = f"{round(percent,1)}% of {outX} are {outY}."
            # Fix acronym casing
            for term in ["NSCLC", "EGFR", "KRAS", "HER2", "ALK"]:
                answer = answer.replace(term.lower(), term)
            return answer
        # Simple percentage of total for one filter
        if "female" in q_lower:
            count = len(df[df['Gender']=="Female"])
            pct = (count/len(df)*100) if len(df)>0 else 0
            return f"{round(pct,1)}% of volunteers are female."
        if "male" in q_lower:
            count = len(df[df['Gender']=="Male"])
            pct = (count/len(df)*100) if len(df)>0 else 0
            return f"{round(pct,1)}% of volunteers are male."
        if "other" in q_lower or "non-binary" in q_lower:
            count = len(df[df['Gender']=="Other"])
            pct = (count/len(df)*100) if len(df)>0 else 0
            return f"{round(pct,1)}% of volunteers are of Other gender."
        if "diab" in q_lower:
            count = len(df[df['Diabetes']=="Yes"])
            pct = (count/len(df)*100) if len(df)>0 else 0
            return f"{round(pct,1)}% of volunteers have diabetes."
        if "pregnan" in q_lower:
            count = len(df[df['Pregnant']=="Yes"])
            pct = (count/len(df)*100) if len(df)>0 else 0
            return f"{round(pct,1)}% of volunteers are pregnant."
    # Average/median age
    if "average age" in q_lower or "mean age" in q_lower or "avg age" in q_lower:
        if total == 0:
            return "No volunteers to calculate an average age."
        avg_age = round(sub_df['Age'].mean(), 1)
        return f"The average age of the volunteers is {avg_age} years."
    if "median age" in q_lower:
        if total == 0:
            return "No volunteers to calculate median age."
        median_age = int(sub_df['Age'].median())
        return f"The median age of the volunteers is {median_age} years."
    # Youngest/Oldest
    if "youngest" in q_lower or "minimum age" in q_lower:
        if total == 0:
            return "No volunteers in this group."
        min_age = sub_df['Age'].min()
        youngest = sub_df[sub_df['Age'] == min_age]['Name'].tolist()
        if len(youngest) == 1:
            ctx = f" from {region_filter}" if region_filter else ""
            return f"The youngest volunteer{ctx} is {youngest[0]}, who is {min_age} years old."
        else:
            return f"The youngest volunteers are " + ", ".join(youngest) + f", all {min_age} years old."
    if "oldest" in q_lower or "maximum age" in q_lower:
        if total == 0:
            return "No volunteers in this group."
        max_age = sub_df['Age'].max()
        oldest = sub_df[sub_df['Age'] == max_age]['Name'].tolist()
        if len(oldest) == 1:
            ctx = f" from {region_filter}" if region_filter else ""
            return f"The oldest volunteer{ctx} is {oldest[0]}, who is {max_age} years old."
        else:
            return f"The oldest volunteers are " + ", ".join(oldest) + f", all {max_age} years old."
    # Gender distribution
    if ("male and female" in q_lower) or ("male & female" in q_lower) or (re.search(r'\bmale\b', q_lower) and re.search(r'\bfemale\b', q_lower) and "how many" in q_lower):
        male_count = len(sub_df[sub_df['Gender']=="Male"])
        female_count = len(sub_df[sub_df['Gender']=="Female"])
        other_count = len(sub_df[sub_df['Gender']=="Other"])
        if other_count > 0:
            return f"There are {male_count} male, {female_count} female, and {other_count} other volunteers."
        else:
            return f"There are {male_count} male and {female_count} female volunteers."
    if "gender distribution" in q_lower or "gender breakdown" in q_lower:
        total_sub = len(sub_df)
        if total_sub == 0:
            return "No volunteers in this group."
        male_count = len(sub_df[sub_df['Gender']=="Male"])
        female_count = len(sub_df[sub_df['Gender']=="Female"])
        other_count = len(sub_df[sub_df['Gender']=="Other"])
        parts = []
        if male_count: parts.append(f"{male_count} male ({round(male_count/total_sub*100,1)}%)")
        if female_count: parts.append(f"{female_count} female ({round(female_count/total_sub*100,1)}%)")
        if other_count: parts.append(f"{other_count} other ({round(other_count/total_sub*100,1)}%)")
        return "Gender distribution: " + ", ".join(parts) + "."
    # Region with most volunteers
    if "which region has the most" in q_lower or ("region" in q_lower and "most volunteers" in q_lower):
        if len(df) == 0:
            return "No volunteer data available."
        counts = df['Region'].value_counts()
        top_region = counts.index[0]; top_count = counts.iloc[0]
        return f"{top_region} has the most volunteers ({top_count})."
    if ("each region" in q_lower or "by region" in q_lower) and "how many" in q_lower:
        if len(df) == 0:
            return "No volunteer data."
        counts = df['Region'].value_counts().to_dict()
        parts = [f"{reg}: {cnt}" for reg, cnt in counts.items()]
        return "Volunteers by region - " + "; ".join(parts) + "."
    # List volunteers
    if "list" in q_lower or q_lower.startswith("name") or "who are" in q_lower or "names of" in q_lower:
        if total == 0:
            return "No volunteers meet that criteria."
        names = sub_df['Name'].tolist()
        if total == 1:
            return f"The volunteer is {names[0]}."
        else:
            return "The volunteers are " + ", ".join(names) + "."
    # Yes/No any volunteers
    if q_lower.startswith("are there any"):
        if total > 0:
            desc = q[len("are there any"):].strip()
            return f"Yes, there {'is' if total==1 else 'are'} {total} {desc}."
        else:
            return "No, there are none."
    # Default count
    if "how many" in q_lower or "number of" in q_lower or "count" in q_lower or "total" in q_lower or q_lower.endswith("volunteers"):
        if total == 0:
            return "There are no volunteers matching that criteria."
        answer = f"There {'is' if total==1 else 'are'} {total} "
        answer += (gender_filter.lower() + " ") if gender_filter else ""
        answer += f"volunteer{'s' if total!=1 else ''}"
        details = []
        if condition_filter:
            cond_names = condition_filter if isinstance(condition_filter, list) else [condition_filter]
            cond_text = " and ".join(cond_names)
            details.append(f"with {cond_text}")
        if biomarker_filter:
            if biomarker_filter.endswith('-'):
                marker = biomarker_filter[:-1]; text = f"{marker} negative"
            else:
                marker = biomarker_filter[:-1] if biomarker_filter.endswith('+') else biomarker_filter
                text = f"{marker} positive" if biomarker_filter.endswith('+') else marker
            if condition_filter or region_filter:
                details.append(f"who are {text}")
            else:
                details.append(f"with {text} biomarker")
        if region_filter:
            details.append(f"from {region_filter}")
        if stage_filter:
            details.append(f"at stage {' or '.join([s.upper() for s in stage_filter])}")
        if diabetes_filter is True:
            if any(part.startswith("with") for part in details):
                # append to last 'with' part if exists
                combined = False
                for i, part in enumerate(details):
                    if part.startswith("with") and "and" not in part:
                        details[i] = part + " and diabetes"
                        combined = True
                        break
                if not combined:
                    details.append("with diabetes")
            else:
                details.append("with diabetes")
        if diabetes_filter is False:
            if any(part.startswith("with") for part in details):
                combined = False
                for i, part in enumerate(details):
                    if part.startswith("with") and "and" not in part:
                        details[i] = part + " and no diabetes"
                        combined = True
                        break
                if not combined:
                    details.append("with no diabetes")
            else:
                details.append("with no diabetes")
        if pregnant_filter is True:
            details.append("who are pregnant")
        if pregnant_filter is False:
            details.append("who are not pregnant")
        detail_str = (" " + " ".join(details)) if details else ""
        return answer + detail_str + "."
    return "I'm sorry, I cannot determine the answer to that question."

# UI Layout
st.title("AI Clinical Trial Volunteer Recruiter")
tab1, tab2 = st.tabs(["TechVitals Admin", "Medical Company"])

with tab1:
    st.header("TechVitals Admin")
    st.write("Upload a CSV of volunteers or use the sample dataset provided. You can filter the volunteers table by age, gender, and region, and ask natural language questions about the dataset.")
    # CSV file uploader
    file_csv = st.file_uploader("Upload Volunteer CSV", type=['csv'])
    if file_csv is not None:
        st.session_state.volunteers_df = pd.read_csv(file_csv)
    df_vol = st.session_state.volunteers_df
    if df_vol.empty:
        st.info("No volunteer data available.")
    else:
        # Filter controls
        genders = ["All"] + sorted(df_vol['Gender'].dropna().unique().tolist())
        regions = ["All"] + sorted(df_vol['Region'].dropna().unique().tolist())
        min_age_val = int(df_vol['Age'].min()) if not df_vol.empty else 0
        max_age_val = int(df_vol['Age'].max()) if not df_vol.empty else 100
        age_range = st.slider("Age Range", min_value=min_age_val, max_value=max_age_val, value=(min_age_val, max_age_val))
        gender_choice = st.selectbox("Gender", options=genders, index=0)
        region_choice = st.selectbox("Region", options=regions, index=0)
        # Apply filters to DataFrame
        filtered_df = df_vol.copy()
        filtered_df = filtered_df[(filtered_df['Age'] >= age_range[0]) & (filtered_df['Age'] <= age_range[1])]
        if gender_choice != "All":
            filtered_df = filtered_df[filtered_df['Gender'] == gender_choice]
        if region_choice != "All":
            filtered_df = filtered_df[filtered_df['Region'] == region_choice]
        st.subheader("Volunteer List")
        st.dataframe(filtered_df.reset_index(drop=True))
        # Q&A about dataset
        st.subheader("Ask Questions about Volunteers")
        with st.form(key="dataset_qa_form"):
            q1 = st.text_input("Ask a question about the volunteer dataset:", placeholder="e.g., How many volunteers have diabetes?")
            submit_q1 = st.form_submit_button("Ask")
        if submit_q1 and q1:
            answer1 = answer_question_about_df(q1, df_vol)
            st.session_state.dataset_qa_history.append((q1, answer1))
        if len(st.session_state.dataset_qa_history) > 5:
            st.session_state.dataset_qa_history = st.session_state.dataset_qa_history[-5:]
        if st.session_state.dataset_qa_history:
            st.markdown("**Last 5 Q&A:**", unsafe_allow_html=True)
            hist_html = "<div style='max-height: 150px; overflow-y:auto; padding:5px; border:1px solid #CCC;'>"
            for ques, ans in st.session_state.dataset_qa_history:
                hist_html += f"<p><b>Q:</b> {ques}</p><p><b>A:</b> {ans}</p><hr>"
            hist_html += "</div>"
            st.markdown(hist_html, unsafe_allow_html=True)

with tab2:
    st.header("Medical Company")
    st.write("Upload a trial criteria PDF or fill out the trial criteria form to find matching volunteers for the clinical trial. You can view a summary of the criteria, see the eligible volunteers, and ask questions about the matched volunteers.")
    mode = st.radio("Input Trial Criteria:", ["Upload PDF", "Fill Form"], index=0)
    if mode != st.session_state.mode:
        st.session_state.active_criteria = False
        st.session_state.matched_df = None
        st.session_state.inclusion_points = []
        st.session_state.exclusion_points = []
    st.session_state.mode = mode
    criteria = {}
    if mode == "Upload PDF":
        pdf_file = st.file_uploader("Upload Trial Criteria PDF", type=['pdf'])
        if pdf_file is not None:
            # Only parse new PDF if a different file is uploaded
            if st.session_state.last_pdf_name != pdf_file.name:
                pdf_bytes = pdf_file.getvalue()
                criteria, inc_points, exc_points = parse_trial_pdf(pdf_bytes)
                st.session_state.inclusion_points = inc_points
                st.session_state.exclusion_points = exc_points
                st.session_state.matched_df = filter_volunteers(st.session_state.volunteers_df, criteria)
                st.session_state.active_criteria = True
                st.session_state.last_pdf_name = pdf_file.name
    else:  # Fill Form
        with st.form(key="criteria_form"):
            age_min = st.number_input("Minimum Age", min_value=0, max_value=120, value=18)
            age_max = st.number_input("Maximum Age", min_value=0, max_value=120, value=100)
            cond_input = st.text_input("Required Condition (e.g., type of cancer):", "")
            biomarker_input = st.text_input("Required Biomarker Status (e.g., EGFR+):", "")
            stages_allowed = st.multiselect("Allowed Disease Stages:", ['I','II','III','IV'], default=['I','II','III','IV'])
            gender_allowed = st.selectbox("Allowed Gender:", ["Any", "Male", "Female"], index=0)
            exclude_diab = st.checkbox("Exclude volunteers with diabetes")
            exclude_preg = st.checkbox("Exclude pregnant volunteers")
            submit_form = st.form_submit_button("Find Eligible Volunteers")
        if submit_form:
            criteria = {
                'min_age': int(age_min), 'max_age': int(age_max),
                'conditions': [cond_input] if cond_input else [],
                'biomarker': biomarker_input if biomarker_input else None,
                'stages': stages_allowed if stages_allowed else [],
                'gender': gender_allowed, 
                'exclude_diabetes': exclude_diab, 'exclude_pregnant': exclude_preg
            }
            # Normalize any condition acronyms spacing
            criteria['conditions'] = [c.strip() for c in criteria['conditions'] if c.strip()]
            criteria['stages'] = [s.strip().upper() for s in criteria['stages']]
            st.session_state.inclusion_points = []
            st.session_state.exclusion_points = []
            st.session_state.matched_df = filter_volunteers(st.session_state.volunteers_df, criteria)
            st.session_state.active_criteria = True
    if st.session_state.active_criteria:
        st.subheader("Trial Criteria Summary")
        # Display criteria bullet points
        if mode == "Upload PDF" and st.session_state.inclusion_points:
            # Show inclusion and exclusion lists from PDF
            if st.session_state.inclusion_points:
                st.markdown("**Inclusion Criteria:**")
                for pt in st.session_state.inclusion_points:
                    st.markdown(f"- {pt}")
            if st.session_state.exclusion_points:
                st.markdown("**Exclusion Criteria:**")
                for pt in st.session_state.exclusion_points:
                    st.markdown(f"- {pt}")
        else:
            # Construct summary from criteria dict (manual form)
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
        # Eligible volunteers table and filters
        eligible_df = st.session_state.matched_df.copy().reset_index(drop=True) if st.session_state.matched_df is not None else pd.DataFrame()
        st.subheader("Eligible Volunteers")
        if eligible_df.empty:
            st.write("No volunteers meet the trial criteria.")
        else:
            # Filters for eligible volunteers table
            regions2 = ["All"] + sorted(eligible_df['Region'].dropna().unique().tolist())
            gender2 = ["All"] + sorted(eligible_df['Gender'].dropna().unique().tolist())
            stage2 = ["All"] + sorted(eligible_df['DiseaseStage'].dropna().unique().tolist())
            min_age2 = int(eligible_df['Age'].min()); max_age2 = int(eligible_df['Age'].max())
            age_range2 = st.slider("Age Range", min_value=min_age2, max_value=max_age2, value=(min_age2, max_age2))
            gender_choice2 = st.selectbox("Gender", options=gender2, index=0)
            region_choice2 = st.selectbox("Region", options=regions2, index=0)
            stage_choice2 = st.selectbox("Disease Stage", options=stage2, index=0)
            filtered_matched = eligible_df.copy()
            filtered_matched = filtered_matched[(filtered_matched['Age'] >= age_range2[0]) & (filtered_matched['Age'] <= age_range2[1])]
            if gender_choice2 != "All":
                filtered_matched = filtered_matched[filtered_matched['Gender'] == gender_choice2]
            if region_choice2 != "All":
                filtered_matched = filtered_matched[filtered_matched['Region'] == region_choice2]
            if stage_choice2 != "All":
                filtered_matched = filtered_matched[filtered_matched['DiseaseStage'] == stage_choice2]
            st.write(f"Found {len(filtered_matched)} volunteer(s) meeting the criteria.")
            st.dataframe(filtered_matched.reset_index(drop=True))
            # Download button for CSV of eligible volunteers
            csv_data = filtered_matched.to_csv(index=False)
            st.download_button("Download Eligible Volunteers CSV", data=csv_data, file_name="Eligible_Volunteers.csv", mime="text/csv")
            # Download button for PDF criteria (if applicable)
            if mode == "Upload PDF" and pdf_file is not None:
                st.download_button("Download Criteria PDF", data=pdf_file.getvalue(), file_name="Trial_Criteria.pdf", mime="application/pdf")
            # Q&A about matched volunteers
            st.subheader("Ask Questions about Matched Volunteers")
            with st.form(key="matched_qa_form"):
                q2 = st.text_input("Ask a question about the matched volunteers:", placeholder="e.g., How many are from Delhi?")
                submit_q2 = st.form_submit_button("Ask")
            if submit_q2 and q2:
                # Use full matched set (before dropdown filters) for Q&A context
                full_matched_df = st.session_state.matched_df if st.session_state.matched_df is not None else filtered_matched
                answer2 = answer_question_about_df(q2, full_matched_df)
                st.session_state.history_matched.append((q2, answer2))
            if len(st.session_state.history_matched) > 5:
                st.session_state.history_matched = st.session_state.history_matched[-5:]
            if st.session_state.history_matched:
                st.markdown("**Last 5 Q&A:**", unsafe_allow_html=True)
                hist2_html = "<div style='max-height: 150px; overflow-y:auto; padding:5px; border:1px solid #CCC;'>"
                for ques, ans in st.session_state.history_matched:
                    hist2_html += f"<p><b>Q:</b> {ques}</p><p><b>A:</b> {ans}</p><hr>"
                hist2_html += "</div>"
                st.markdown(hist2_html, unsafe_allow_html=True)

