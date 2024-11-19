# Customer Churn Prediction Model

This project aims to build a machine learning model to predict customer churn based on various features in the customer dataset.

## Overview

The script loads customer churn data, preprocesses it, trains a Random Forest Classifier, and evaluates its performance against a baseline model using a Dummy Classifier. The model is designed to handle both numeric and categorical features, and it evaluates performance on different customer segments based on their IDs.

## Features

- **Data Loading**: Loads customer churn data from a CSV file.
- **Feature Engineering**: Identifies numeric and categorical features.
- **Model Training**: Uses Random Forest Classifier and a Dummy Classifier as a baseline.
- **Evaluation**: Evaluates model performance using accuracy and classification report.
- **Segmentation**: Evaluates the model on older and newer customers based on `customer_id`.
- **Model Persistence**: Saves the trained models to disk for future use.

## Requirements

Ensure that you have the necessary libraries installed. You can install the required packages using pip:

```bash
pip install -r requirements.txt
