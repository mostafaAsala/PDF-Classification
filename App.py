import json
from fastapi import FastAPI, APIRouter, HTTPException, UploadFile, File, Form
from fastapi.templating import Jinja2Templates
import numpy as np
from pydantic import BaseModel
from utils.Model import TrainingModels
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import os
import shutil
import uuid
from typing import List
import warnings
from utils.pdf_module import extract_text_from_pdf_Production

warnings.filterwarnings('ignore')

app = FastAPI()

# Define the request body
class URLRequest(BaseModel):
    """
        This Pydantic model defines the request body structure for the /predict/ endpoint.
    Args:
        BaseModel (_type_): _description_
    """
    url: str

# Define the request body for CSV file path
class CSVRequest(BaseModel):
    """
        This Pydantic model defines the request body structure for handling CSV file paths.
    Args:
        BaseModel (_type_): _description_
    """
    csv_path: str

# Define the prediction endpoint
router = APIRouter()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize the TrainingModels class
model_instance = TrainingModels()
model_instance.load_models(prefix="",date="")

@router.post("/predict/")
async def predict(request: URLRequest): 
    """
    o	Receives a URL in the request body.
    o	Uses the model_instance to predict based on the PDF URL.
    o	Returns a JSON response with the prediction and confidence.


    Args:
        request (URLRequest): Request body containing the URL of the PDF.

    Raises:
        Exception: any

    Returns:
        _type_: dict
    """
    try:
        # Predict using the model
        status,prediction = model_instance.predict_pdf(request.url)
        print(status,prediction)
        if status==1:
            for element in prediction:
                if isinstance(prediction[element][0], np.float32):
                    prediction[element]=float(prediction[element][0])
                else:
                    prediction[element]=prediction[element][0]
            prediction['URL'] = request.url
            print(prediction)
            return prediction
        else:
            return json.dumps({'Error': prediction})
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/predict_file/")
async def predict_file(file: UploadFile = File(...),preExtractedText: bool = Form(...)): 
    """
    o	Handles file uploads (CSV, XLS, XLSX).
    o	Saves the uploaded file to a temporary location.
    o	Reads the file into a DataFrame.
    o	Uses model_instance to predict based on the DataFrame.


    Args:
        file (UploadFile, optional): The uploaded file. Defaults to File(...).

    Raises:
        HTTPException: Unsupported file type
        Exception: any

    Returns:
        _type_: FileResponse
    """
    try:
        temp_dir = "/tmp"
        temp_file_path = f"/tmp/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        #result_df = model_instance.predict_Files(df)
        if preExtractedText and 'text' in df.columns:
            result_df = model_instance.predict_text_df(df)
        else: 
            result_df = model_instance.predict_Files(df)
        output_csv_path = f"/tmp/{uuid.uuid4()}.csv"
        result_df.to_csv(output_csv_path, index=False)
        return FileResponse(output_csv_path, filename="result.csv", media_type="text/csv")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract_file/")
async def extract_file(file:UploadFile=File(...)):
    try:
        temp_dir = "/tmp"
        os.makedirs(temp_dir,exist_ok=True)
        temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()))
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        url_column = 'URL'
        if url_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"'{url_column}' column not found in the uploaded file")

        
        async def generate_results():
            for index, row in df.iterrows():
                batch=row['URL']
                batch_results = extract_text_from_pdf_Production(batch)
                for index, result in batch_results.iterrows():
                    try:
                        yield json.dumps(result.to_dict()) + "\n"
                    except Exception  as e:
                        print(f"Owner error: {e}")
        return StreamingResponse(generate_results(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        pass
    

@router.post("/predict_url_file/")
async def predict_url_file(file: UploadFile = File(...), number: int = Form(...)):
    """
    o	Handles file uploads (CSV, XLS, XLSX).
    o	Saves the uploaded file to a temporary location.
    o	Reads the file into a DataFrame.
    o	Sample the data by Number.
    o	Uses model_instance to predict sample based on the DataFrame.
    o	Returns the sample result as a table in the api


    Args:
        file (UploadFile, optional): The uploaded file. Defaults to File(...).
        number (int, optional): Number of lines extracted at once. Defaults to Form(...).

    Raises:
        HTTPException: Unsupported file type
        HTTPException: URL column not found in the uploaded file

    Returns:
        _type_: List of dicts

    Yields:
        _type_: Dict
    """
    try:
        temp_dir = "/tmp"
        os.makedirs(temp_dir, exist_ok=True)
        temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()))

        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        if file.filename.endswith('.csv'):
            df = pd.read_csv(temp_file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(temp_file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")

        url_column = 'URL'
        if url_column not in df.columns:
            raise HTTPException(status_code=400, detail=f"'{url_column}' column not found in the uploaded file")

        

        async def generate_results():
            batch_size = number
            for start in range(0, len(df), batch_size):
                if batch_size> len(df):batch_size = len(df)-1
                if start + batch_size>len(df)-1:
                    batch = df.iloc[start:len(df)-1]
                
                else:
                    batch = df.iloc[start:start + batch_size]
                
                batch_results = model_instance.predict_Files(batch)
                for index, result in batch_results.iterrows():
                    result['Index'] = index
                    print(result)
                    
                    y=result.to_dict()
                    try:
                        x= json.dumps(y)
                        yield json.dumps(result.to_dict()) + "\n"
                    except Exception  as e:
                        print(f"Owner error: {e}")

        return StreamingResponse(generate_results(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@router.post("/extract_text_from_pdf/")
async def extract_text_from_pdf(request: URLRequest):
    """
    o	Extract text from a pdf 
    o	Using OCR if the pdf doesn’t have text

    Args:
        request (URLRequest): the URL link for the pdf

    Returns:
        _type_: Dict
    """
    print("extract text")
    try:
        pages,text = extract_text_from_pdf_Production(request.url,pages=10000)
        return JSONResponse(content={"text": text,"pages":pages})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/")
async def root():
    """
    o	Serves the index.html file from the templates directory.

    Returns:
        _type_: FileResponse
    """
    return FileResponse("templates/index.html")

# Include the router in the main app
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
