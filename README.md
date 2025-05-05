# Heart Disease Prediction Using Machine Learning and Deep Neural Networks

## Project Overview

This project aims to predict heart disease using machine learning algorithms and a deep neural network model. The dataset used for this project contains medical data of patients, and the goal is to predict the likelihood of heart failure based on various health indicators.

### Algorithms Used:
- **K-Nearest Neighbors (KNN)**
- **Support Vector Machine (SVM)**
- **Logistic Regression**
- **Neural Network (using TensorFlow/Keras)**

## Objective

The main objective of this project is to predict the possibility of heart disease in patients by analyzing various health parameters such as age, cholesterol levels, blood pressure, etc. The dataset is used to train machine learning models to classify patients into two categories: those with heart disease and those without.

## Dataset

The dataset contains the following features:
- **age**: Age of the patient
- **sex**: Gender of the patient
- **cp**: Chest pain type
- **trestbps**: Resting blood pressure
- **chol**: Serum cholesterol
- **fbs**: Fasting blood sugar
- **restecg**: Resting electrocardiographic results
- **thalach**: Maximum heart rate achieved
- **exang**: Exercise induced angina
- **oldpeak**: Depression induced by exercise relative to rest
- **slope**: Slope of the peak exercise ST segment
- **ca**: Number of major vessels colored by fluoroscopy
- **thal**: Thalassemia
- **target**: Presence or absence of heart disease (1 = disease, 0 = no disease)

## Steps Involved

1. **Data Preprocessing**:
   - Data cleaning and handling missing values.
   - Feature scaling using StandardScaler.
   
2. **Model Selection**:
   - Tried multiple models including **KNN**, **SVM**, **Logistic Regression**, and **Neural Network**.
   - Hyperparameter tuning using **GridSearchCV** for optimal parameters.
   
3. **Model Evaluation**:
   - Evaluated models using metrics like **accuracy**, **precision**, **recall**, and **F1-score**.
   - Confusion matrix and classification report were generated for each model.

4. **Hyperparameter Tuning**:
   - Optimized learning rate using **Learning Rate Schedulers**.
   - Best hyperparameters were selected for each algorithm through **GridSearchCV**.

5. **Final Model**:
   - Final models were trained on the dataset and their performance was compared.

## Results

The following models were evaluated:

1. **KNN (K-Nearest Neighbors)**:
   - Accuracy: 98%
   - Precision: 0.98, Recall: 0.98, F1-Score: 0.98

2. **SVM (Support Vector Machine)**:
   - Accuracy: 98%
   - Precision: 0.99, Recall: 0.97, F1-Score: 0.98

3. **Logistic Regression**:
   - Accuracy: 85%
   - Precision: 0.87, Recall: 0.81, F1-Score: 0.84

4. **Neural Network (Keras)**:
   - Accuracy: 98%
   - Precision: 0.97, Recall: 0.98, F1-Score: 0.98

## Setup and Installation

To run this project locally, follow these steps:

1. Clone this repository:

   ```bash
   git clone https://github.com/khashayard/heart-disease-prediction.git
