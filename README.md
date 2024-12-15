# Flight Arrival Delay Prediction Using PySpark

## Overview

This project aims to predict flight arrival delays using a regression model. The dataset used is provided by the U.S. Department of Transportation's (DOT) Bureau of Transportation Statistics, which tracks the on-time performance of domestic flights operated by major air carriers. The prediction model is built using PySpark, leveraging its distributed computing capabilities to handle and process large datasets efficiently.
The goal of this project is to create a scalable machine learning pipeline that predicts flight arrival delays by analyzing historical flight data, including details on flight times, airports, and carriers.

## Dataset
The dataset used in this project consists of three primary files:
- flights.csv: Contains information about each flight, such as flight times, delays, and cancellations.
- airports.csv: Contains details about the airports, including their location and IATA codes.
- airlines.csv: Contains information about the airlines, including their IATA codes.

Note on the Airline Table:

In this project, we do not use the airlines.csv file, as it only includes the full names of airline companies, which are not relevant for predicting flight delays. Therefore, we will be focusing only on the flights.csv and airports.csv datasets for feature extraction and model training.

You can safely ignore the airlines.csv file when running the project.

## Project Workflow

1. Data Preprocessing
The first step in the project is to preprocess the data, including:
- Loading the Data: The datasets are loaded into PySpark DataFrames.
- Feature Selection: Relevant features for predicting arrival delays are selected from the flights.csv dataset.
- Joining Tables: The flights.csv dataset is merged with the airports.csv dataset to enrich the data with additional airport information.
- Missing Value Handling: Missing values are checked and imputed, ensuring no more than 10% of records are dropped.
- Final Dataset Preparation: After preprocessing, the dataset is saved to persistent storage (e.g., DBFS or Google Drive) and then reloaded for further use.
- Train-Test Split: The final dataset is split into training and testing sets (70% training, 30% testing).

2. Building the ML Pipeline
A machine learning pipeline is constructed with the following steps:
- Encoding Categorical Features: Categorical features (e.g., airport and airline codes) are encoded into numeric values using techniques like StringIndexer.
- Scaling Numerical Features: Numerical features (e.g., distances, times) are scaled using StandardScaler to standardize the data.
- Feature Vector Assembly: A feature vector is created by combining the encoded categorical features and scaled numerical features using VectorAssembler.
- Linear Regression Model: A linear regression model is built using the preprocessed features and trained on the training dataset.

3. Model Evaluation
After training the model, its performance is evaluated using the Mean Absolute Error (MAE) metric:
- Training MAE: The MAE is calculated on the training dataset.
- Testing MAE: The MAE is calculated on the testing dataset, with the goal of achieving an MAE below 25 minutes.

## Dataset Download

Due to the large size of the datasets, they are hosted on Google Drive. Please download the following files and place them in the `data/` directory:

- [flights.csv](https://drive.google.com/file/d/1FvNW68prpJvuNDxlI-6Rke5Pml-RmmRO/view?usp=drivesdk)
- [airports.csv](https://drive.google.com/file/d/1Qbgdx4UuYYkOK2inwSUW8VCXUie5hqcb/view?usp=drivesdk)
- [airlines.csv](https://drive.google.com/file/d/1hLzxNORaUBIiFxZ2be8V7bqOIL1N4h8P/view?usp=drivesdkl)

## Getting Started
1. Clone the repository:
```
git clone https://github.com/NgChunMan/Flight-Arrival-Delay-Prediction.git
cd Flight-Arrival-Delay-Prediction
```

2. After cloning the repository, open the config.json file and replace the flights_path and airports_path fields with the correct paths to the flights.csv and airports.csv files on your local machine or cloud storage.
   
3. Install the required dependencies using pip:
```
pip install -r requirements.txt
```

3. Run the Python script:
```
python flight_delay_prediction.py
```
This will execute the full pipeline, from loading the data to evaluating the model performance.

## Evaluation Metrics

The model's performance is evaluated based on the Mean Absolute Error (MAE). The goal is to achieve the following:
- Training MAE: To check how well the model fits the training data.
- Testing MAE: To ensure that the model generalizes well to unseen data, with a target of MAE < 25 minutes.
