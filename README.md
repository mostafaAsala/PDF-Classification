# PDF Classification

## Overview
This project involves a comprehensive pipeline for PDF document classification, including data preprocessing, model training, and prediction. The project structure is organized into various directories for data, models, results, and utilities.


## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

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
## Project Details

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

### FastAPI Application

This FastAPI application provides endpoints for making predictions based on URLs and uploaded files.

#### Endpoints

- **`/predict/`**: POST - Predicts based on a PDF URL.
- **`/predict_file/`**: POST - Handles file uploads and predictions.
- **`/predict_url_file/`**: POST - Sample and predict based on uploaded file.
- **`/extract_text_from_pdf/`**: POST - Extract text from a PDF.
- **`/`**: GET - Serves the index.html file.

#### Application Setup

```python
from fastapi import FastAPI

app = FastAPI()
```

---

This README provides an overview of the project, its structure, and details on how to use it. For more specific information, please refer to the individual module sections.




























Here is a README file converted from your document:

---

# Project Documentation

## Project Introduction

This project involves a comprehensive pipeline for PDF document classification, including data preprocessing, model training, and prediction. The project structure is organized into various directories for data, models, results, and utilities.

### Project Structure

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

### Project Details

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

### Installation

1. Install all required packages:
   ```
   pip install -r requirements.txt
   ```
2. Set environment variables:
   - `poopler_Path`: Path to Poppler.
   - `tesseract_path`: Path to Tesseract.

### Training

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

### FastAPI Application

This FastAPI application provides endpoints for making predictions based on URLs and uploaded files.

#### Endpoints

- **`/predict/`**: POST - Predicts based on a PDF URL.
- **`/predict_file/`**: POST - Handles file uploads and predictions.
- **`/predict_url_file/`**: POST - Sample and predict based on uploaded file.
- **`/extract_text_from_pdf/`**: POST - Extract text from a PDF.
- **`/`**: GET - Serves the index.html file.

#### Application Setup

```python
from fastapi import FastAPI

app = FastAPI()
```

---

This README provides an overview of the project, its structure, and details on how to use it. For more specific information, please refer to the individual module sections.
