import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
import numpy as np
import torch
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz, process
import math
from collections import defaultdict


country_dict = {}
coutnries = []
with open('full_country_code_to_name.json', 'r') as file:
    country_dict = json.load(file)
    countries = [i.lower() for i in country_dict.values()] 

model = SentenceTransformer('all-MiniLM-L6-v2')


# Download NLTK data
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')



## --------- Skills similarity -----------

def get_skill_embedding(skill_text, model, min_words_for_bigrams=3):
    """
    Generate a vector representation for a skill.
    If the skill text is long (>= min_words_for_bigrams), break it into bigrams and average their embeddings.
    Otherwise, use the embedding of the full skill text.
    """
    words = skill_text.split()
    if len(words) >= min_words_for_bigrams:
        # Create bigrams from the words
        bigrams = [" ".join(bigram) for bigram in zip(words, words[1:])]
        bigram_embeddings = model.encode(bigrams)
        # Average the embeddings to get a single representation
        return np.mean(bigram_embeddings, axis=0)
    else:
        return model.encode(skill_text)

def one_way_similarity_with_bigrams(source_skills, target_skills, model):
    """
    Compute the 'one-way' similarity from source_skills -> target_skills.
    For each skill in source_skills, compute its embedding and find the maximum
    cosine similarity with any skill in target_skills. Return the average of these maxima.
    
    Example usage:
        job_to_resume = one_way_similarity_with_bigrams(job_skills, resume_skills, model)
        resume_to_job = one_way_similarity_with_bigrams(resume_skills, job_skills, model)
        final_score = (job_to_resume + resume_to_job) / 2
    """
    if not source_skills:
        # If there are no source skills, treat as trivially matched
        return 1.0
    if not target_skills:
        # If there are no target skills to match against
        return 0.0
    
    target_embeddings = [get_skill_embedding(skill, model) for skill in target_skills]
    
    max_similarities = []
    for skill in source_skills:
        skill_emb = get_skill_embedding(skill, model)
        sims = cosine_similarity([skill_emb], target_embeddings)
        max_similarities.append(sims.max())
    
    return np.mean(max_similarities)

def two_way_skill_similarity(job_skills, resume_skills, model):
    """
    Computes the average of job->resume similarity and resume->job similarity.
    This can provide a more balanced measure if you want to account for 'extra'
    skills in either the job description or the resume.
    """
    if not job_skills and not resume_skills:
        return 1.0  # Both empty => trivial match
    if not job_skills or not resume_skills:
        return 0.0  # One side is empty => no match

    job_to_resume = one_way_similarity_with_bigrams(job_skills, resume_skills, model)
    resume_to_job = one_way_similarity_with_bigrams(resume_skills, job_skills, model)
    return (job_to_resume + resume_to_job) / 2.0

def compute_skill_score(
    required_skills, 
    preferred_skills, 
    resume_skills, 
    model, 
    required_weight=2.0, 
    preferred_weight=1.0, 
    use_two_way=True
):
    """
    Compute an overall skill score from required and preferred skills.
    - required_skills and preferred_skills come from the job.
    - resume_skills come from the candidate's resume.
    - required_weight: how strongly to weight required skills.
    - preferred_weight: how strongly to weight preferred skills.
    - use_two_way: whether to use two-way skill similarity or only job->resume.
    
    Returns a combined skill similarity score (0.0 to 1.0).
    """
    if use_two_way:
        required_sim = two_way_skill_similarity(required_skills, resume_skills, model) if required_skills else 1.0
        preferred_sim = two_way_skill_similarity(preferred_skills, resume_skills, model) if preferred_skills else 1.0
    else:
        required_sim = one_way_similarity_with_bigrams(required_skills, resume_skills, model) if required_skills else 1.0
        preferred_sim = one_way_similarity_with_bigrams(preferred_skills, resume_skills, model) if preferred_skills else 1.0
    
    # Weighted average
    # E.g. if required_weight=2.0, preferred_weight=1.0 => total weight=3.0
    total_weight = required_weight + preferred_weight
    weighted_score = (required_sim * required_weight + preferred_sim * preferred_weight) / total_weight
    
    return round(float(weighted_score),2)

def match_certifications(job_certs, resume_certs):
    ## We might want to make this better by instead of matching complete certification names, 
    ## Similarity match between names of certifications
    """
    Quick example for certifications.
    If the job lists certain required certifications and the resume does not have them,
    penalize. Similarly, if the job has optional certs, provide a partial match.
    This is simplified logic you can expand as needed.
    """
    if not job_certs:
        return 1.0  # No required cert => trivial match
    
    # For illustration, let's do a simple matching approach: exact string matches
    # If you want partial matches, you could embed them or do partial ratio string matching
    job_certs_lower = [c.lower().strip() for c in job_certs]
    resume_certs_lower = [c.lower().strip() for c in resume_certs]
    
    # Weighted approach: required vs. optional – if your JD splits them that way
    # For now, let's assume all certifications in "job_certs" are required
    missing = 0
    for cert in job_certs_lower:
        if cert not in resume_certs_lower:
            missing += 1
    
    # If any required certification is missing, let’s reduce the score
    # This is a simplistic approach: if any are missing, it can be 0 or partial
    if missing == 0:
        return 1.0
    else:
        # Example: if some are missing, reduce score proportionally
        coverage_ratio = (len(job_certs_lower) - missing) / len(job_certs_lower)
        return coverage_ratio  # in [0..1]

def match_experience_level(job_exp_level, resume_total_months):
    # If no experience is specified in the job, treat it as a trivial match
    if 'not specified' in job_exp_level.lower():
        return 1.0
    
    # 1) Try to match a range like "3-5 years"
    match_range = re.search(r'(\d+)\s*-\s*(\d+)', job_exp_level)
    if match_range:
        # e.g. job_exp_level = "3-5 years"
        min_years = int(match_range.group(1))  # e.g. 3
        # upper_years = int(match_range.group(2)) # e.g. 5, if you want to use or check that
        min_months = min_years * 12
    else:
        # 2) Otherwise, try to match a single number like "3 years" or "3+ years"
        match_single = re.search(r'(\d+)', job_exp_level)
        if match_single:
            min_years = int(match_single.group(1))
            min_months = min_years * 12
        else:
            # 3) Fallback, we can’t find any digits => no specified minimum
            min_months = 0
    
    # Compare candidate’s experience to the min months required
    if resume_total_months >= min_months:
        return 1.0
    else:
        # Partial credit for partially meeting the requirement
        # (You can tweak if you prefer a sharp 0 for failing to meet min.)
        return resume_total_months / float(min_months) if min_months else 0


## --------- Location Helpers ---------
def extract_location_from_description(location_str: str, country_list: list) -> str:
    """
    Returns the first matched country name from the description.
    """
    location_str_lower = location_str.lower()
    
    for country in country_list:
        # Use word boundaries to avoid partial matches like "land" in "Finland"
        if re.search(rf"\b{re.escape(country)}\b", location_str_lower):
            return country  # First match found

    return None  # No country found

def extract_country_code(phone_number: str) -> str:
    cleaned = re.sub(r"[()\s\-]", "", phone_number)
    match = re.match(r'^(?:\+|00)(\d{1,4})', cleaned)
    return match.group(1) if match else None


def check_location_match(job_location, resume_location, resume_phone):
    location = None
    job_loc = extract_location_from_description(job_location, countries)
    
    if job_loc is None:
        return -1

    # Try to get country from phone
    if resume_phone:
        extracted_country_code = extract_country_code(resume_phone)
        if extracted_country_code and extracted_country_code in country_dict:
            location = country_dict[extracted_country_code]

    # Fallback to resume location
    if location is None and resume_location:
        location = resume_location

    if location is None:
        return 0

    # Check if job_loc is mentioned in resume_location (case-insensitive)
    if job_loc in location.lower():
        return 1
    else:
        return 0

  
# ---------- Education Helpers ----------

def load_qs_ranking_data(file_path):
    """
    Loads QS ranking data from an Excel file.
    Expected columns: "University Name", "QS Rank".
    """
    try:
        qs_ranking_df = pd.read_csv(file_path)

        # Ensure consistent column names
        if "University Name" not in qs_ranking_df.columns or "QS_Rank" not in qs_ranking_df.columns:
            raise ValueError("The Excel file must contain 'University Name' and 'QS Rank' columns.")
        return qs_ranking_df
    except Exception as e:
        print(f"Error loading QS ranking data: {e}")
        return None
    
    
def calculate_log_scores(qs_ranking_df):
    """
    Adds a log-based score column to the QS ranking DataFrame.
    """
    max_rank = qs_ranking_df["QS_Rank"].max()
    
    def log_score(rank):
        if rank <= 0:
            return 0  # Handle edge case where rank is undefined
        return (np.log(max_rank) - np.log(rank)) / np.log(max_rank)
    
    qs_ranking_df["Log Score"] = qs_ranking_df["QS_Rank"].apply(log_score)
    return qs_ranking_df


def find_university_match(candidate_university, qs_ranking_df):
    """
    Matches a candidate's university name to the closest match in the QS ranking data.
    1. Try exact match.
    2. Try prefix match.
    3. If no close enough match, return None.
    """
    qs_universities = qs_ranking_df["University Name"].dropna().tolist()

    # Step 1: Exact match
    if candidate_university in qs_universities:
        return candidate_university

    # Step 2: Prefix match (strict start of string match)
    for uni in qs_universities:
        if uni.lower().startswith(candidate_university.lower()):
            return uni

    # Step 3: Fuzzy match only if similarity is very high
    best_match, match_score = process.extractOne(candidate_university, qs_universities, scorer=fuzz.token_sort_ratio)

    # Avoid matching unrelated names with weak similarity
    if match_score >= 90:
        return best_match
    else:
        return None

    
def get_log_score(matched_university, qs_ranking_df):
    """
    Retrieves the log-based score for a matched university.
    Returns 0 if no match is found.
    """
    if matched_university is not None:
        score = qs_ranking_df.loc[qs_ranking_df["University Name"] == matched_university, "Log Score"].values
        return score[0] if len(score) > 0 else 0
    else:
        return 0    
    

def get_education_level_weight(level):
    level = level.lower()
    if "phd" in level or "doctorate" in level:
        return 1.0
    elif "master" in level:
        return 0.9
    elif "bachelor" in level:
        return 0.7
    elif "diploma" in level:
        return 0.5
    return 0.5  # default weight

def get_education_score(university_name, education_level):
    if university_name is None:
        return 0
    qs_ranking_file = "./qs.csv"
    qs_ranking_df = load_qs_ranking_data(qs_ranking_file)
    qs_ranking_df = calculate_log_scores(qs_ranking_df)
   
    matched_university = find_university_match(university_name, qs_ranking_df)
    edu_level = get_education_level_weight(education_level)
    education_score = (edu_level) + get_log_score(matched_university, qs_ranking_df)    
    
    return round(float(education_score), 2)


def calculate_education_similarity(job_education, resume_education):
    """
    Calculate similarity score between job education requirements and resume education.
    Considers degree level, field of study, and university prestige (QS ranking).
    
    Args:
        job_education (str): Education requirement from job description
        resume_education (list): List of education entries from resume
        qs_ranking_df (pd.DataFrame): DataFrame containing QS university rankings
    
    Returns:
        float: Similarity score between 0 and 1
    """
    if not job_education or 'not specified' in job_education.lower():
        return 1.0  # No education requirement means perfect match
    
    if not resume_education:
        return 0.0  # No education in resume means no match
    
    # Extract degree level from job requirement
    job_degree_level = None
    degree_keywords = {
        'bachelor': 'Bachelor',
        'master': 'Master',
        'phd': 'PhD',
        'doctorate': 'PhD',
        'diploma': 'Diploma',
        'degree': 'Bachelor'  # Default to Bachelor if just "degree" is mentioned
    }
    
    job_education_lower = job_education.lower()
    for keyword, level in degree_keywords.items():
        if keyword in job_education_lower:
            job_degree_level = level
            break
    
    if not job_degree_level:
        return 1.0  # If we can't determine required level, assume perfect match
    
    # Get highest degree from resume
    resume_degrees = [edu.get('level_of_education', '') for edu in resume_education]
    resume_degrees = [d for d in resume_degrees if d]  # Remove empty values
    
    if not resume_degrees:
        return 0.0
    
    # Map degree levels to numeric values for comparison
    degree_levels = {
        'Diploma': 1,
        'Bachelor': 2,
        'Masters': 3,
        'PhD': 4
    }
    
    job_level = degree_levels.get(job_degree_level, 0)
    resume_level = max(degree_levels.get(level, 0) for level in resume_degrees)
    
    # Calculate degree level match (0.6 weight)
    if resume_level >= job_level:
        degree_score = 1.0
    else:
        degree_score = resume_level / job_level
    
    # Calculate field of study match (0.4 weight)
    field_score = 0.0
    if job_education:
        # Extract field of study from job requirement
        field_keywords = ['in', 'of', 'field']
        job_field = None
        for keyword in field_keywords:
            if keyword in job_education_lower:
                job_field = job_education.split(keyword)[-1].strip()
                break
        
        if job_field:
            # Check if any resume education matches the field
            for edu in resume_education:
                resume_field = edu.get('field_of_study', '')
                if resume_field and job_field.lower() in resume_field.lower():
                    field_score = 1.0
                    break
    
    
    # Combine scores with weights
    final_score = (degree_score * 0.6) + (field_score * 0.4)
    
    
    return final_score






# ------ Experience Helper --------------    

# Function to preprocess text (stopword removal + lemmatization)
def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    if not text:  # Handle None or empty strings
        return ""
    
    text = text.lower().translate(str.maketrans('', '', string.punctuation))  # Lowercase & remove punctuation
    words = word_tokenize(text)  # Tokenize
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]  # Lemmatize & remove stopwords
    
    return " ".join(words)

# Function to compute BERT similarity for job titles
def bert_similarity(title1, title2):
    embeddings = model.encode([title1, title2], convert_to_tensor=True)
    return torch.nn.functional.cosine_similarity(embeddings[0].unsqueeze(0), embeddings[1].unsqueeze(0)).item()

# Function to compute TF-IDF + one-way cosine similarity for summaries
def tfidf_similarity(job_summary, experience_summaries):
    valid_summaries = [summary for summary in experience_summaries if summary]  # Remove empty summaries

    if not valid_summaries:  # If no valid summaries, return zero scores
        return np.zeros(len(experience_summaries))

    vectorizer = TfidfVectorizer()
    corpus = [job_summary] + valid_summaries  # Job description first
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    job_vector = tfidf_matrix[0]  # Job description vector
    experience_vectors = tfidf_matrix[1:]  # Experience vectors
    
    # Compute cosine similarity
    scores = cosine_similarity(job_vector, experience_vectors)[0]

    # Assign zero similarity to missing summaries
    full_scores = []
    index = 0
    for summary in experience_summaries:
        if summary:
            full_scores.append(scores[index])
            index += 1
        else:
            full_scores.append(0.0)  # Missing summary gets zero similarity

    return full_scores

# Function to rank experiences based on weighted similarity
def rank_experiences(job_desc, experiences):
    job_title = job_desc['position']
    
    # Create JD summary (concatenate skills & soft skills)
    job_summary = " ".join(job_desc["required_skills"] + job_desc["preferred_skills"] + job_desc["soft_skills"])
    job_summary = preprocess_text(job_summary)  # Preprocess JD summary
    
    exp_titles = [exp["title"] for exp in experiences]
    exp_summaries = [preprocess_text(exp["summary"]) if exp.get("summary") else None for exp in experiences]

    # Compute similarities
    title_similarities = [bert_similarity(job_title, title) for title in exp_titles]
    summary_similarities = tfidf_similarity(job_summary, exp_summaries)

    # Compute final weighted score
    final_scores = [
        (0.75 * title_sim + 0.25 * summary_sim) if summary is not None else title_sim
        for title_sim, summary_sim, summary in zip(title_similarities, summary_similarities, exp_summaries)
    ]

    # Sort experiences by similarity score
    ranked_experiences = sorted(zip(experiences, final_scores), key=lambda x: x[1], reverse=True)
    for exp, score in zip(experiences, final_scores):
        exp["score"] = score

    # Step 2: Filter out experiences with score <= 0.4
    filtered_experiences = [exp for exp in experiences if exp["score"] > 0.4]

    # Step 3: Compute weighted score and total duration
    weighted_sum = 0.0
    total_duration = 1
    for exp in filtered_experiences:
        weighted_sum += exp["score"] *  exp.get("duration_months", 1)
        total_duration += exp.get("duration_months", 0)



    return round(weighted_sum/total_duration, 2), total_duration


def meets_experience_requirement(experience_string, candidate_years):
    """
    Extracts the minimum years of experience required from the experience_string by searching for numbers 
    that are associated with the words 'year' or 'years'. Then, returns True if candidate_years is greater 
    than or equal to the extracted minimum, otherwise returns False.
    
    Parameters:
        experience_string (str): A string containing experience information.
        candidate_years (int): The candidate's years of experience.
    
    Returns:
        bool: True if candidate_years meets or exceeds the minimum required years, False otherwise.
    
    Raises:
        ValueError: If no valid numeric experience requirement is found.
    """
    # Regex to match a number (optionally followed by a range or a plus sign) that is immediately followed by "year" or "years"
    pattern = re.compile(r'(\d+)(?:\s*[-+]\s*(\d+))?\s*(?=years?\b)', re.IGNORECASE)
    matches = pattern.findall(experience_string)
    
    if not matches:
        return True
    
    # Extract the minimum from each match (the first captured group)
    min_years_list = [int(match[0]) for match in matches]
    min_required = min(min_years_list)
    
    return candidate_years >= min_required

# -------- Weights Adjustment Helper ---------------

def calculate_dynamic_weights(jd_json):
    """
    Comprehensive dynamic weight calculation with intelligent redistribution rules.
    Handles all possible missing criteria cases following these redistribution priorities:
    
    Redistribution Rules:
    1. Location (-1):
       - 50% to relevant_experience
       - 30% to technical_skills
       - 20% to education
    2. Missing Certifications:
       - 100% to technical_skills
    3. Missing Soft Skills:
       - 70% to technical_skills
       - 30% to relevant_experience
    4. Missing Education:
       - 60% to relevant_experience
       - 40% to technical_skills
    5. Missing Experience:
       - 100% to technical_skills
    6. Missing Technical Skills (edge case):
       - Distribute equally among remaining criteria
    """
    # Base weights (sum = 1.0)
    base_weights = {
        'technical_skills': 0.40,
        'relevant_experience': 0.25,
        'education': 0.15,
        'certifications': 0.08,
        'soft_skills': 0.07,
        'location': 0.05
    }
    
    

    # Determine presence of each criteria
    present = {
        'technical_skills': bool(jd_json.get('required_skills') or jd_json.get('preferred_skills')),
        'relevant_experience': bool(jd_json.get('experience_level')),
        'education': bool(jd_json.get('highest_level_of_education')),
        'certifications': bool(jd_json.get('certifications')),
        'soft_skills': bool(jd_json.get('soft_skills')),
        'location': 'location' in jd_json and jd_json['location'] != -1
    }

    # Apply redistribution rules in priority order
    redistribution_plan = defaultdict(float)

    # Rule 1: Handle location (-1 case)
    if 'location' in jd_json and jd_json['location'] == -1:
        if present['relevant_experience']:
            redistribution_plan['relevant_experience'] += base_weights['location'] * 0.5
        if present['technical_skills']:
            redistribution_plan['technical_skills'] += base_weights['location'] * 0.3
        if present['education']:
            redistribution_plan['education'] += base_weights['location'] * 0.2
        present['location'] = False

    # Rule 2: Missing certifications
    if not present['certifications']:
        if present['technical_skills']:
            redistribution_plan['technical_skills'] += base_weights['certifications']
        present['certifications'] = False

    # Rule 3: Missing soft skills
    if not present['soft_skills']:
        if present['technical_skills']:
            redistribution_plan['technical_skills'] += base_weights['soft_skills'] * 0.7
        if present['relevant_experience']:
            redistribution_plan['relevant_experience'] += base_weights['soft_skills'] * 0.3
        present['soft_skills'] = False

    # Rule 4: Missing education
    if not present['education']:
        if present['relevant_experience']:
            redistribution_plan['relevant_experience'] += base_weights['education'] * 0.6
        if present['technical_skills']:
            redistribution_plan['technical_skills'] += base_weights['education'] * 0.4
        present['education'] = False

    # Rule 5: Missing experience
    if not present['relevant_experience']:
        if present['technical_skills']:
            redistribution_plan['technical_skills'] += base_weights['relevant_experience']
        present['relevant_experience'] = False

    # Apply redistributions
    for criterion, amount in redistribution_plan.items():
        base_weights[criterion] += amount

    # Rule 6: Handle edge case where technical skills are missing
    if not present['technical_skills']:
        remaining_criteria = [c for c, exists in present.items() if exists]
        if remaining_criteria:
            weight_to_distribute = base_weights['technical_skills']
            per_criteria = weight_to_distribute / len(remaining_criteria)
            for c in remaining_criteria:
                base_weights[c] += per_criteria
            present['technical_skills'] = False

    # Calculate total present weight
    total_present_weight = sum(
        weight for criterion, weight in base_weights.items()
        if present[criterion]
    )

    # Normalize weights
    if total_present_weight > 0:
        adjusted_weights = {
            criterion: (base_weights[criterion] / total_present_weight)
            if present[criterion] else 0
            for criterion in base_weights
        }
    else:
        # Ultimate fallback - equal weights
        adjusted_weights = {c: 1.0/len(base_weights) for c in base_weights}

    # Final verification
    total = sum(adjusted_weights.values())
    if abs(total - 1.0) > 0.0001:
        adjusted_weights = {k: v/total for k, v in adjusted_weights.items()}

    return adjusted_weights



# -------- Score Calculation Helper ---------------

def calc_score(result, weights):
    """
    Calculate the final score based on the results and weights.
    """
    
    overall_score = 0.0
    for key, value in result.items():
        overall_score += value * weights[key]
    
    return overall_score

## ------- Combined Pipeline ---------------

def pipeline(resume_data, job_data):    

    # For the job:
    required_skills = job_data.get("required_skills", [])
    preferred_skills = job_data.get("preferred_skills", [])
    soft_skills_job = job_data.get("soft_skills", [])
    job_certs = job_data.get("certifications", [])
    job_exp_level = job_data.get("experience_level", "Not specified")
    
    # For the resume:
    resume_tech_skills = resume_data.get("technical_skills", [])  # or "technical_skills"
    resume_soft_skills = resume_data.get("soft_skills", [])
    resume_certs = resume_data.get("certifications", [])
    resume_experience_months = resume_data.get("total_experience_months", 0)
    resume_university = resume_data["education"][0]["university"] if len(resume_data["education"]) else None
    education_level = resume_data["education"][0]["level_of_education"] if len(resume_data["education"]) else None
    # 1. Compute technical skill score (required + preferred)
    tech_score = compute_skill_score(
        required_skills=required_skills,
        preferred_skills=preferred_skills,
        resume_skills=resume_tech_skills,
        model=model,
        required_weight=2.0,
        preferred_weight=1.0,
        use_two_way=False  # or False, depending on your preference
    )

    # 2. Compute soft skill score using two-way or one-way
    soft_skill_score = round(float(one_way_similarity_with_bigrams(soft_skills_job, resume_soft_skills, model)),2)
    
    # 3. Certifications match
    cert_score = match_certifications(job_certs, resume_certs)
    
    # 4. Location match
    location_score = check_location_match(job_data["location"], resume_data["Location"], resume_data["phone"])


    # 5. Education University Score
    education_uni_score = get_education_score(resume_university, education_level)



    # 6. Education Match Score
    education_match_score = calculate_education_similarity(job_data["highest_level_of_education"], resume_data["education"])


    # Education Overall Score
    education_score = round(0.4 * education_uni_score + 0.6 * education_match_score , 2)

    # 7. Experience Score
    exp_score, exp_duration = rank_experiences(job_data, resume_data["experience"])
    years_of_experience = math.ceil(exp_duration/12)
    exp_match_score = meets_experience_requirement(job_exp_level, years_of_experience)
    
    experience_score = 0.5 * exp_score + 0.5 * int(exp_match_score)


    # 8. Calculate Weights    
    weights = calculate_dynamic_weights(job_data)
    result =  {
        "technical_skills": float(tech_score),
        "soft_skills": float(soft_skill_score),
        "certifications": float(cert_score),
        "relevant_experience": float(experience_score),
        "education": float(education_score),
        "location": float(location_score),
    }
    
    result["overall_score"] = round(float(calc_score(result, weights)),4)
    
    return weights, result



## Loop

def evaluate_data(job_data, resume_data):
    output = {}
    for key, value in resume_data.items():
        weights, res = pipeline(value, job_data)
        output[key] = res
    return output