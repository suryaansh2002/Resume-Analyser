from fastapi import FastAPI,  UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from jd_helper import *
from resume_helper import *
from evaluation_helper import *

from pydantic import BaseModel
import zipfile
import io
import PyPDF2

# Create a FastAPI instance
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods
    allow_headers=["*"],  # Allows all headers
)

class JobDescription(BaseModel):
    job_description: str = None


class Resumes(BaseModel):
    job_title: str = None
    resumes: dict = None
    

class ResumesJDCombined(BaseModel):
    job_data: dict = None
    resumes: dict = None

# Define a simple route
@app.get("/")
def health_check():
    return {"status": "Healthy"}


@app.post("/job_description")
def job_description(job_description: JobDescription):
    return jd_to_dict(job_description)


@app.post("/upload_zip/")
async def extract_pdf_text(file: UploadFile = File(...)):
    # Ensure the uploaded file is a zip file
    if file.content_type != "application/zip":
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a zip file.")

    # Read the entire file content into memory
    file_content = await file.read()
    pdf_texts = {}
    
    try:
        # Open the zip file from the in-memory bytes
        with zipfile.ZipFile(io.BytesIO(file_content)) as z:
            # Loop through all files in the zip
            for filename in z.namelist():
                formatted_filename = filename.split('/')[-1]
                if filename.endswith(".pdf"):
                    with z.open(filename) as pdf_file:
                        pdf_bytes = pdf_file.read()
                        try:
                            # Use PyPDF2 to read and extract text from the PDF
                            pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
                            text = ""
                            for page in pdf_reader.pages:
                                extracted_text = page.extract_text() or ""
                                text += extracted_text
                            pdf_texts[formatted_filename] = text
                        except Exception as e:
                             print (f"Error reading PDF {filename}: {e}")
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid zip file.")
    
    return  pdf_texts


@app.post("/resumes")
def resumes(data: Resumes):
    return all_resume_to_json(data.job_title, data.resumes)


@app.post("/evaluate")
def resumes(data: ResumesJDCombined):
    return evaluate_data(data.job_data, data.resumes)
