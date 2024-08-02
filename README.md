# Hyperparameter Tuning for Early Diagnosis of Parkinson’s Disease Using Big Data

This project focuses on using big data and hyperparameter tuning techniques to improve the early diagnosis of Parkinson's Disease. The analysis is performed using various datasets related to Parkinson’s Disease, processed and analyzed using Apache Spark.

## Introduction

This project aims to utilize various datasets related to Parkinson’s Disease to develop models that can aid in its early diagnosis. The project leverages the power of Apache Spark for big data processing and focuses on hyperparameter tuning to enhance model performance.

## Datasets

The datasets used in this project include:

- Patient Status
- Demographics
- Family History
- MDS-UPDRS Parts I, II, III
- MoCA
- STAI
- GDS
- QUIP
- SCOPA-AUT
- SFT
- REM RBD
- Epworth
- HVLT
- LNS
- Neuro-Cranial
- SDM
- Benton

Each dataset is stored in a CSV file and is read and processed using PySpark.

## Setup and Dependencies

To run this project, you need to have the following dependencies installed:

- Python
- PySpark
- Jupyter Notebook

### Installation

1. Clone this repository.
2. Install the required Python packages.
   ```bash
   pip install pyspark


## Data Loading and Preprocessing
The data is loaded and preprocessed using PySpark. The main steps include:

Reading Datasets: Each dataset is read from a CSV file and registered as a temporary table in Spark.
Filtering and Cleaning: Unwanted columns are dropped, and specific filtering criteria are applied.
Imputation: Missing values are imputed using linear interpolation.

## Hyperparameter Tuning
Hyperparameter tuning is performed to optimize the model parameters for better performance. The process involves:

Defining the parameter grid.
Using cross-validation to evaluate different parameter combinations.
Selecting the best model based on performance metrics.
