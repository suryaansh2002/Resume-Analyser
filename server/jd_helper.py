
import json
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from datetime import datetime

def extract_job_description_info(job_description_text, llm_model="gemma3"):
    print(f"Extracting JOB DESCRIPTION INFO")
    """
    Extracts relevant information from a job description text and returns it as a JSON dictionary.

    Args:
        job_description_text (str): The job description text.
        llm_model (str): Name of the Ollama model to use.

    Returns:
        dict: Extracted information in JSON format, or None if an error occurs.
    """
    try:
        llm = ChatOllama(model=llm_model, temperature=0.8)

        template = """
        Extract the following information from the job description text provided, and format it as a JSON dictionary.

        Job Description Text:
        {job_description_text}

        Information to extract:
        {{
            "company": "Name of the company",
            "position": "Job title/position name",
            "highest_level_of_education": "Highest level and degree of education",
            "required_skills": ["List of required technical or hard skills"],
            "soft_skills": ["List of required soft skills"],
            "preferred_skills": ["List of preferred skills"],
            "experience_level": "Experience level required (e.g., Senior, Junior, Mid-level)",
            "location": "Job location",
            "certifications": ["List of required or preferred certifications"]
        }}

        Instructions:
        1. Focus on identifying technical skills, soft skills, and preferred skills explicitly mentioned in the job description.

        JSON:
        """
        
        prompt = PromptTemplate(template=template, input_variables=["job_description_text"])
        llm_chain = LLMChain(prompt=prompt, llm=llm)

        response = llm_chain.run(job_description_text=job_description_text)
        print(f"Response: {response}")
        try:
            return response.strip()
        except json.JSONDecodeError:
            return None

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

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


def jd_to_dict(job_description_text, llm_model="gemma3"):
    """
    Extracts relevant information from a job description text and returns it as a JSON dictionary.

    Args:
        job_description_text (str): The job description text.
        llm_model (str): Name of the Ollama model to use.

    Returns:
        dict: Extracted information in JSON format, or None if an error occurs.
    """
    try:    
        dict_string = extract_job_description_info(job_description_text, llm_model)
        return preprocess_json_string_to_dict(dict_string)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
