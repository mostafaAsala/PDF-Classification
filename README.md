# PDF Classification

## Overview
This project focuses on classifying PDF documents related to electronic components into the following categories:

1. Datasheet
2. Environmental
3. Manual and Guide
Others
## Table of Contents
1. [Installation](#installation)
2. [Data Preparation](#data-preparation)
3. [Training the Model](#training-the-model)
4. [Evaluation](#evaluation)
5. [Results](#results)
6. [Contributing](#contributing)
7. [License](#license)

## Installation
To get started, clone the repository and install the required dependencies.

```bash
git clone https://github.com/your-username/repository-name.git
cd repository-name
pip install -r requirements.txt
```

## Data Preparation
1. **Data Directory:**  
   Place your dataset in the `data/` directory. If the directory doesn't exist, create it.

   ```bash
   mkdir data
   ```

2. **Data Format:**  
   Ensure your data is in the correct format. The model expects the data in the following format:
   - **Features:** 600 features per sample.
   - **Labels:** 10 possible classes.

   Example file structure:
   ```
   data/
   ├── train.csv  # Training data
   └── test.csv   # Test data
   ```

3. **Data Preprocessing:**  
   Before training, ensure your data is preprocessed correctly. The model pipeline automatically handles tokenization, vectorization, and feature selection.

## Training the Model
To train the model, use the following command:

```bash
python train.py
```

This script will:
- Load the data from the `data/` directory.
- Preprocess the data using the pipeline specified in the code.
- Train the [Random Forest/XGBoost/Other Model] on the training data.
- Save the trained model to the `models/` directory.

**Important:**  
Ensure that the training data (`train.csv`) is correctly placed in the `data/` directory before running the script.

## Evaluation
After training, evaluate the model using:

```bash
python evaluate.py
```

This script will:
- Load the trained model from the `models/` directory.
- Evaluate it on the test dataset (`test.csv`).
- Print out the performance metrics, including recall, precision, F1 score, etc.

## Results
The results of the model evaluation will be saved in the `results/` directory. You can find detailed reports and visualizations of the model's performance here.

## Contributing
If you'd like to contribute to this project, please fork the repository and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

You can customize this template further based on your specific project needs.
