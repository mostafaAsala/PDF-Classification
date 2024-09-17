# PDF Classification

## Overview
This project involves a comprehensive pipeline for PDF document classification, including data preprocessing, model training, and prediction. The project structure is organized into various directories for data, models, results, and utilities.


## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Project files Structure](#project-structure)
4. [Project files Details](#project-files-details)
5. [Training the Model](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Documentation Summary](#documentation-summary)
9. [API](#fastapi-application)

## Installation

1. Install all required packages:
   ```
   pip install -r requirements.txt
   ```
2. Set environment variables:
   - `poopler_Path`: Path to Poppler.
   - `tesseract_path`: Path to Tesseract.


## Data Preparation
1. **Data Directory:**  
   Place your dataset in the `data/` directory. If the directory doesn't exist, create it.

   ```bash
   mkdir data
   ```
  
3. **Data Format:**
   you have three stages, the **links.csv** contains only the links and classifications, when running training the extracted text is put into column in **debug_data_out.csv** then split into trining and testing
   Ensure your data is in the correct format. The model expects the data in the following format:
   - **links:** links of pdf
   - **text:** text column of pdf "not needed in links file"
   - **classification:** 4 possible classes.

## Project Structure
   
   ```
   Project
   │
   └── data/
   │   ├── train/
   │   │   └── train.csv  # Training data
   │   └── test/
   │       └── test.csv   # Test data
   │   └── debug_data_out.csv 
   │   └── link.csv
   │
   └── models/
   │   └── RandomForest_model.pkl
   │   └── Label_encoders.pkl
   │   └── ...
   │
   └── Results/
   │   └── ...
   │
   └── static/
   │   └── images/
   │   └── CSS/
   │
   └── templates/
   │   └── index.html
   │
   └── utils/
   │   └── constants.py
   │   └── mainTest.py
   │   └── Model.py
   │   └── pdf_module.py
   │   └── readFile.py
   │   └── txtFileManagement.py
   │
   └── App.py
   └── requirements.txt
   └── RunLocal.bat
   └── train.py
   ```
## Project files Details

- **Data Folder**: Contains data in different stages, with `links.csv` for links and classifications. Extracted text is added to `debug_data_out.csv`, then split into training (`train/train.csv`) and testing (`test/test.csv`). The model manages the stage to begin based on arguments passed to `train.py`.

- **Expected Data Columns**:
  - `links`: PDF links
  - `text`: Text column of PDF (not needed in the links file)
  - `classification`: 4 possible classes

- **Models**: Stores the result models, encoders, pipelines, and vectorizers.
- **Results**: Stores the prediction results of the last operation.
- **Static**: Stores the static images and styles of the web app.
- **Templates**: Contains `index.html`, the frontend of the web app.
- **Utils**: Various utility scripts and configurations.
- **Requirements.txt**: Lists all the packages required for the project.
- **RunLocal.bat**: Script to launch the web app locally.

## Training

To train a model, add data to the `data` folder as described, then run the following command:

```
python train.py --no_eval_only --load_split --Preload_text --sampling=1 --type=orig
```

#### Arguments

- **Boolean Arguments**:
  - `--eval_only`: Load a trained model and evaluate testing data.
  - `--no_eval_only`: Train the model from the start. (default)
  - `--load_split`: Load train and test data from a previous split.
  - `--no_load_split`: Load full data and split it into train and test. (default)
  - `--Preload_text`: Load data from file with preloaded text column. (default)
  - `--no_Preload_text`: Load file without text and load text from the `text_files` folder.

- **Integer Arguments**:
  - `--sampling`: Controls sampling technique (`1` for oversampling, `0` for no sampling, `-1` for undersampling).

- **String Arguments**:
  - `--type`: Specify the type of model to train. Possible values:
    - `orig`: Original model for classifying PDFs into {Datasheet, Environmental, Manual and Guide, Others}.
    - `Datasheet`: Distinguish between {Datasheet, Application Brief, Product Brief, Package Manual, Guide}.
    - `Environmental`: Check for separate environmental types with multiple output columns (True/False).
    - `Others`: Distinguish between {Life_Cycle, News, Application Brief, Product Brief, Package Drawing, PCN, Package Brief, Others}.


This command will:
- Load the data from the `data/` directory.
- Preprocess the data using the pipeline specified in the code.
- Train the [Random Forest/XGBoost/Other Model] on the training data.
- Save the trained model to the `models/` directory.

**Important:**  
Ensure that the training data (`links.csv`) or (`debug_data_out.csv`) is correctly placed in the `data/` directory before running the script.

## Evaluation
the model automatically evaluate the testing data placed in `data/test/test.csv` 
to evaluate only without training you can use command `--eval_only` while runing `train.py` 

This script will:
- Load the trained model from the `models/` directory.
- Evaluate it on the test dataset (`test.csv`).
- Print out the performance metrics, including recall, precision, F1 score, etc.

## Results
The results of the model evaluation will be saved in the `results/` directory. You can find detailed reports and visualizations of the model's performance here.



## Documentation Summary


### PDF Module

The `pdf_module.py` is a utility module for handling various PDF-related tasks, including reading, validating, downloading, processing, and extracting text from PDF files. It is particularly useful for automating text extraction from large collections of PDF documents.

#### Key Functions

1. **`read_text(file_path: str) -> str`**: Reads the content of a text file.
2. **`is_Searchable(pdf_path: str, top_margin: float, bottom_margin: float, thresh: int) -> bool`**: Checks if a PDF contains searchable text.
3. **`is_valid_pdf_link(url: str) -> bool`**: Validates if a URL is a valid PDF link.
4. **`download_pdf(url: str, save_path: str) -> bool`**: Downloads a PDF from a URL.
5. **`process_pdf(pdf_bytes: bytes, max_pages: int) -> dict`**: Processes a PDF byte stream to extract text.
6. **`extract_text_from_pdf_Production(pdf_path: str, top_margin: float, bottom_margin: float) -> str`**: Extracts text from a PDF.
7. **`get_links(file_path: str) -> pd.DataFrame`**: Reads a CSV file and returns valid PDF links.
8. **`get_all_files(pdf_path: str, datasheet_path: str, use_Exist: bool) -> pd.DataFrame`**: Consolidates PDF links from two CSV files.
9. **`extract_text_from_pdf(pdf_path: bytes, txt_path: str, top_margin: float, bottom_margin: float) -> bool`**: Extracts text from a PDF file.
10. **`data_extract3(dataLabel_List: list, exclude_list: list, start: int, finish: int) -> None`**: Extracts data from a collection of PDF files.

### Text File Management Module

This module provides utility functions for handling and processing CSV files and text files.

#### Key Functions

1. **`remove_o(directory: str) -> None`**: Renames files by removing '.0' from filenames.
2. **`export_id(directory: str, filename: str) -> None`**: Exports IDs of files to a CSV file.
3. **`get_ids_from_debug(source_path: str, dest_path: str) -> None`**: Extracts 'pdf_url' from a CSV file.
4. **`update_classification(id_source: str, data_classification: str) -> None`**: Updates document classification.
5. **`save_csv_to_files(path: str, dir: str) -> None`**: Saves text content from a CSV file to text files.
6. **`join_Files(file1: str, file2: str, res: str) -> None`**: Merges two CSV files.
7. **`check_csv(path: str) -> None`**: Prints the columns length and value counts.
8. **`remove_duplicates(file: str) -> None`**: Removes duplicate rows from a CSV file.

### Model

#### `TextPreprocessor` Class

Responsible for text preprocessing, including tokenization, stopword removal, lemmatization, stemming, and removal of URLs, numbers, and emojis.

#### `TrainingModels` Class

Handles the training and evaluation of multiple classifiers on preprocessed text data.

### Usage

#### Preprocessing Text

```python
preprocessor = TextPreprocessor(n_jobs=4)
preprocessed_text = preprocessor.transform(['Sample text to preprocess.'])
print(preprocessed_text)
```

#### Training Models

```python
trainer = TrainingModels()
trainer.Load_data()
trainer.train()
```

#### Predicting Classes

```python
predictions = trainer.predict(['Sample text to classify.'])
print(predictions)

pdf_url = 'http://example.com/sample.pdf'
pdf_prediction = trainer.predict_pdf(pdf_url)
print(pdf_prediction)
```

# FastAPI Application

This FastAPI application provides endpoints for making predictions based on URLs and uploaded files.

## Endpoints

- **`/predict/`**: POST - Predicts based on a PDF URL.
- **`/predict_file/`**: POST - Handles file uploads and predictions.
- **`/predict_url_file/`**: POST - Sample and predict based on uploaded file.
- **`/extract_text_from_pdf/`**: POST - Extract text from a PDF.
- **`/`**: GET - Serves the index.html file.

## Application Setup

```python
from fastapi import FastAPI

app = FastAPI()
```

## running the web app
```python
python -m uvicorn App:app --reload
```

## API Doccumentation
- **predict pdf**:
   
   - **Endpoint**: /predict/
   - **Method**: POST
   - **Input**: URLRequest data containing a single url string field
        - e.g., {"url": "http://example.com/sample.pdf"}
   - **Output**: JSON response with the predicted class and prediction confidence.
   - **Example Response**:
      ```json
      {
        "model_Prediction": "Datasheet",
        "model_confidence": 0.2689622205537494,
        "URL": "http://download.siliconexpert.com/pdfs2/2022/12/18/0/40/41/502645/zhengs_/auto/mq-7b-ver1_6.pdf"
      }
      ```
- **predict file**:
   
   - **Endpoint**: /predict/
   - **Method**: POST
   - **Input**:
      - **file** string($binary) : file containing header URL to predict its types
      - **preExtractedText** boolean : flag to specify if **text** column exist and is extracted or not
   - **Output**: csv table with predictions
   - **Example Response**:
     - response Body: a csv file contains the result prediction
       
       | Index | URL | model_confidence | model_Prediction | VENDOR_CODE | ONLINE_LINK | PROJECT | Extracted | num_pages | extracted_len | urls | textLenght | numOfPages | Searchable | reliable | lang | Trust_Level flag | 2nd Prediction|
       |------|-------|---------------|--------------------|------------|-------------|----------|-----------|-----------|---------------|------|------------|------------|------------|----------|------|-----------|--------------|
       | 0  | http://offline-example.pdf  | **0.41678303078230416**  | **Datasheet**  | ACL  | https://online-example.pdf  | Ds_Bk  | 1  | 1  | 1983  | http://offline-example.pdf  | 1983  | -  | True  | False  | e  | Low confidence Level  | Product Brief |

     - response header:
         ```curl
          content-disposition: attachment; filename="result.csv" 
          content-length: 4324 
          content-type: text/csv; charset=utf-8 
          date: Tue,17 Sep 2024 07:41:55 GMT 
          etag: "8c79097eebcf7fce262430de968ca5f7" 
          last-modified: Tue,17 Sep 2024 07:42:35 GMT 
          server: uvicorn 
         ```
- **predict file**:
   
   - **Endpoint**: /predict_url_file/
   - **Method**: POST
   - **Input**:
      - **file** string($binary) : file containing header URL to predict its types
      - **number** integer : number of pdf links processed before showing result
   - **Output**: list of JSON response with the predicted class and prediction confidence.
   - **Example Response**:
     - response Body: list of json format responses liske shown for each line in the original csv
       ```json
       {
       "Index": 0,
        "URL": "http://offline-example.pdf",
       "model_confidence": 0.41678303078230416,
       "model_Prediction": "Datasheet",
       "VENDOR_CODE": "-",
       "ONLINE_LINK": "https://online-example.pdf",
       "PROJECT": "Ds_Bk",
       "Extracted": 1,
       "num_pages": 1,
       "extracted_len": 1983,
       "urls": "http://offline-example.pdf",
       "textLenght": 1983,
       "numOfPages": "-",
       "Searchable": true,
       "reliable": false,
       "lang": "e",
       "Trust_Level flag": "Low confidence Level",
       "2nd Prediction": "Product Brief"
       }
       ```
     - response header:
         ```curl
          content-type: application/json 
          date: Tue,17 Sep 2024 08:26:35 GMT 
          server: uvicorn 
          transfer-encoding: chunked 
         ```
- **predict file**:
   
   - **Endpoint**: /extract_text_from_pdf/
   - **Method**: POST
   - **Input**:
      - **URL** string : link for the pdf to be extracted
   - **Output**: csv table with predictions
   - **Example Response**:
     - response Body: json format response with `text` as extracted from pdf
       
       ```json
       {
        "text": "毒性气体传感器 （型号：MQ-7B） 使用说明书 版本号：1.6 实施日期：2021-07-1 郑州炜盛电子科技有限公司 Zhengzhou Winsen Electronic Technology Co.,"
       }
       ```

     - response header:
         ```python
             content-length: 8469 
             content-type: application/json 
             date: Tue,17 Sep 2024 08:34:46 GMT 
             server: uvicorn 
         ```
## Web App 
   - you can use the web app directly with sophisticated interface
![image](https://github.com/user-attachments/assets/6263cbd1-2533-415c-96dd-f8535b847d41)



---

This README provides an overview of the project, its structure, and details on how to use it. For more specific information, please refer to the individual module sections.
