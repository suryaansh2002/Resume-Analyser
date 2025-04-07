


import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re
from dateutil import parser 

from datetime import datetime

def preprocess_resume_text(text):
    """
    Preprocesses resume text, maintaining newlines for better LLM understanding.
    """
    # 1. Remove non-breaking spaces
    text = text.replace('\xa0', ' ')

    # 2. Remove page numbers
    text = re.sub(r'Page \d+ of \d+', '', text)

    # 3. Clean up spacing (but keep newlines)
    lines = text.split('\n')
    cleaned_lines = [' '.join(line.split()) for line in lines]
    text = '\n'.join(cleaned_lines)

    # 4. Standardize bullet points (if necessary)
    text = text.replace('•', '• ')

    return text.strip()

def add_experience_duration(resume_json):
    """Add 'duration_months' to each experience in the resume_json."""
    def calculate_months(start_date, end_date):
        """Calculate total months of experience given two datetime objects.
        This adds 1 to count the starting month as a full month."""
        months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        return months

    total_experience_months = 0

    experiences = []
    for exp in resume_json.get("experience", []):
        if not exp.get("start_date") or not exp.get("end_date"):
            experiences.append(exp)
            continue
        
        start_str = exp.get("start_date", "").strip()
        end_str = exp.get("end_date", "").strip()
        if(end_str.lower() == 'present'):
            end_str = datetime.now().strftime('%m-%Y')
        try:
            start_date = parser.parse(start_str, fuzzy=True)
            end_date = parser.parse(end_str, fuzzy=True)
            months = calculate_months(start_date, end_date)
            total_experience_months += months
            exp['duration_months'] = months
            experiences.append(exp)
            # print(f"Title: {exp.get('title')}")
            # print(f"  From: {start_date.date()} To: {end_date.date()} => {months} month(s)")
        except Exception as e:
            print(f"Error parsing dates for experience '{exp.get('title')}': {e}")
            experiences.append(exp)
    resume_json['experience'] = experiences
    resume_json['total_experience_months'] = total_experience_months

    return resume_json



def extract_resume_info(text, job_title, llm_model="gemma3"):
    """
    Extracts information from a resume PDF using Gemma 3 and returns it as a JSON dictionary.

    Args:
        pdf_path (str): Path to the resume PDF file.
        llm_model (str): Name of the Ollama model to use.

    Returns:
        dict: Extracted information in JSON format, or None if an error occurs.
    """
    try:
        # 2. Setup LLM and Prompt
        llm = ChatOllama(model=llm_model, temperature=0.85)

        template = """
        Extract the following information from the resume text provided, and format it as a JSON dictionary.
        For the job of: {job_title}
        Resume Text:
        {resume_text}

        Information to extract:
        {{
            "name": "Full name of the candidate",
            "phone": "Phone number",
            "Location":"Location of person",            
            "highest_level_of_education":"Highest level anddegree of education",
            "education": [
                {{"degree": "Degree earned", "university": "University name", "graduation_year": "Year of graduation", "level_of_education":"Masters/Bachelors/ etc", "field_of_study":"Field of study", "gpa": "GPA" }},
                //... more education entries as needed
            ],
            
            "experience": [
                {{"title": "Job title", "company": "Company name", float value", "summary": "Job summary", "start_date": "Start date in MM-YYYY", "end_date": "End date in MM-YYYY"}},
                //... more experience entries as needed
            ],


            
            "technical_skills": ["technical_skills1", "technical_skills2", ...],
            "soft_skills": ["soft_skills1", "soft_skills2", ...],
            "certifications": ["certification1", "certification2", ...],
            "languages": ["language1", "language2", ...] # List of languages to the person like English, Hindi, Chinese etc.
            
        }}

        The skills should be extracted from context based on the various experience summaries, certifications done and skills mentioned in the resume.

        Return only the json object in pretty printed form with no other string text before or after it
        """
        
        prompt = PromptTemplate(template=template, input_variables=["resume_text", "job_tite"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)
        # 3. Get LLM response and parse JSON
        response = llm_chain.run(resume_text=text, job_title=job_title)
        try:
            return response.strip()
        except json.JSONDecodeError:
            print(
                f"Error: Could not parse LLM response as JSON. Response was: {response}"
            )
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None
   
def slice_from_braces(s):
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and start < end:
        return s[start:end+1]
    return ''  # return empty string if braces not found properly


def preprocess_json_string_to_dict(markdown_json_str):
    """
    Pre-process a markdown-wrapped JSON string and return the corresponding dictionary.
    
    This function removes markdown code fences (e.g., "```json" and "```") from the input string,
    then parses the remaining JSON string into a Python dictionary.
    
    Args:
        markdown_json_str (str): The JSON string wrapped with markdown code fences.
        
    Returns:
        dict: The parsed JSON content as a Python dictionary.
        
    Raises:
        ValueError: If the JSON is invalid.
    """
    # Remove starting markdown fence if present.
    if markdown_json_str.startswith("```json"):
        markdown_json_str = markdown_json_str[len("```json"):].strip()
    
    # Remove ending markdown fence if present.
    if markdown_json_str.endswith("```"):
        markdown_json_str = markdown_json_str[:-3].strip()
    
    try:
        parsed_dict = json.loads(markdown_json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON data: {e}")
    
    return parsed_dict

    
def resume_to_json(file_name, resume_text, job_title, llm_model="gemma3"):
    """
    Extracts information from a resume PDF and returns it as a JSON dictionary.

    Args:
        file_path (str): Path to the resume PDF file.
        llm_model (str): Name of the Ollama model to use.

    Returns:
        dict: Extracted information in JSON format, or None if an error occurs.
    """
    print(f"Extracting resume information of {file_name}.")
    preprocessed_text = preprocess_resume_text(resume_text)
    response = extract_resume_info(preprocessed_text, job_title, llm_model=llm_model)
    result = slice_from_braces(response)
    response_dict = preprocess_json_string_to_dict(result)
    print(f"Extracted resume information of {file_name}.")
    print(response_dict)
    response_dict = add_experience_duration(response_dict)
    return response_dict
    
    
    

def all_resume_to_json( job_title, resumes, llm_model="gemma3"):
    output = {}
    for file_name, resume_data in resumes.items():
        resume_dict = resume_to_json(file_name, resume_data, job_title, llm_model=llm_model)
        if resume_dict['name']:
            output[resume_dict['name']] = resume_dict
    return output