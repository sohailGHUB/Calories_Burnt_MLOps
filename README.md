# 🔥 Calories Burnt Prediction – End-to-End MLOps on Databricks

> An end-to-end machine learning and MLOps project for predicting calories burned during physical activity using Databricks, MLflow, Unity Catalog, and Databricks Model Serving.

---

## 📑 Project Index

- [📌 Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Technologies Used](#️-technologies-used)
- [📊 Dataset](#-dataset)
- [🔄 MLOps Workflow](#-mlops-workflow)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Development](#2-development)
  - [3. MLflow Experiment Tracking](#3-mlflow-experiment-tracking)
  - [4. Model Logging](#4-model-logging)
  - [5. Model Registry](#5-model-registry)
  - [6. Model Serving](#6-model-serving)
  - [7. Inference](#7-inference)
- [📈 Model Results](#-model-results)
- [🚀 Deployment](#-deployment)
- [📂 Repository Structure](#-repository-structure)
- [▶️ How to Run](#️-how-to-run)
- [🔮 Future Improvements](#-future-improvements)
- [📜 License](#-license)

---

# 📌 Project Overview

This project demonstrates an **end-to-end Machine Learning and MLOps workflow** for predicting calories burned during physical activity using **Azure Databricks, Unity Catalog, MLflow, and Databricks Model Serving**.

The project follows the complete machine learning lifecycle, starting with dataset storage and ingestion, followed by data preparation, model development, experiment tracking, model logging, model registration, model serving, and inference.

The machine learning component focuses on predicting calories burned using exercise and physiological attributes such as:

- Gender
- Age
- Height
- Weight
- Exercise Duration
- Heart Rate
- Body Temperature

Multiple regression algorithms are trained and evaluated during the experimentation stage. **MLflow** is used to track experiments, parameters, metrics, and model artifacts.

After comparing the trained models, **XGBoost** is selected as the best-performing model. The selected model is logged with MLflow, registered in the **Unity Catalog Model Registry**, and deployed using **Databricks Model Serving**.

The deployed model is then tested through a serving endpoint, and its prediction is compared with the local model prediction to validate the deployment.

---

# 🎯 Business Problem

Calories burned during exercise depend on several physiological and activity-related factors. Estimating calorie expenditure manually from these variables can be difficult.

The objective of this project is to build a machine learning regression system capable of predicting calories burned based on an individual's physical characteristics and exercise conditions.

The solution uses historical exercise data to learn the relationship between the input features and calorie expenditure.

### Key Objectives

- Store project datasets using **Unity Catalog Volumes**
- Process and prepare the data using **Azure Databricks**
- Train multiple regression models
- Evaluate and compare model performance
- Track machine learning experiments using **MLflow**
- Select the best-performing model
- Log the selected model using MLflow
- Register the model in **Unity Catalog Model Registry**
- Deploy the registered model using **Databricks Model Serving**
- Perform real-time inference
- Validate the deployed model against the local model

---

# 🏗️ Solution Architecture

The project is built around the following core components:

### 📂 Unity Catalog Volume

The raw project datasets are stored in a **Unity Catalog Volume** within Databricks.

The Volume provides the storage location from which the datasets are accessed during model development.

### 📒 Azure Databricks Notebook

The Databricks Notebook serves as the main development environment for:

- Loading the datasets
- Data preparation
- Exploratory analysis
- Model development
- Model training
- Model evaluation
- MLflow integration
- Model logging
- Model registration
- Serving validation

### 📊 MLflow

MLflow is used to manage the machine learning experimentation and model lifecycle.

It provides:

- Experiment tracking
- Run tracking
- Parameter logging
- Metric logging
- Model artifact logging
- Logged model management

### 🗂️ Unity Catalog Model Registry

The selected model is registered in Unity Catalog after being logged with MLflow.

The registry provides a managed location for the model and its versions before deployment.

### 🚀 Databricks Model Serving

The registered model is deployed using Databricks Model Serving.

The deployed model is exposed through a serving endpoint that can receive prediction requests.

### 🔮 Inference

The serving endpoint accepts the required input features and returns the predicted number of calories burned.

---

# 🖼️ Architecture Diagram

![Calories Burnt MLOps Architecture](01-ARCHITECTURE/calories-burnt-mlops-architecture.png)

The implemented architecture can be summarized as:

```
Raw Datasets
     │
     ▼
Unity Catalog Volume
     │
     ▼
Databricks Notebook
     │
     ▼
Data Preparation & Model Development
     │
     ▼
Multiple ML Models
     │
     ▼
MLflow Experiment Tracking
     │
     ▼
Model Comparison
     │
     ▼
Best Model – XGBoost
     │
     ▼
MLflow Model Logging
     │
     ▼
Unity Catalog Model Registry
     │
     ▼
Databricks Model Serving
     │
     ▼
Inference Endpoint
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Microsoft Azure** | Cloud platform |
| **Azure Databricks** | Machine learning development and execution |
| **Python** | Data processing and machine learning |
| **Pandas** | Dataset loading and manipulation |
| **Scikit-learn** | Machine learning models and evaluation |
| **XGBoost** | Final selected regression model |
| **MLflow** | Experiment tracking and model logging |
| **Unity Catalog** | Data and model governance |
| **Unity Catalog Volumes** | Dataset storage |
| **Unity Catalog Model Registry** | Model registration and version management |
| **Databricks Model Serving** | Model deployment and inference |
| **GitHub** | Source control and project publishing |

---

# ☁️ Azure & Databricks Services Used

| Service / Component | Purpose |
|---|---|
| **Azure Databricks** | Main platform for model development, experimentation, and deployment |
| **Unity Catalog** | Management and governance of project data and models |
| **Unity Catalog Volume** | Storage and access of the raw datasets |
| **MLflow** | Machine learning experiment tracking and model logging |
| **Unity Catalog Model Registry** | Registration and versioning of the selected model |
| **Databricks Model Serving** | Deployment of the registered model for real-time inference |


# 🎯 Objectives

The primary objective of this project is to build and deploy a machine learning model that predicts calories burned during physical activity while implementing the complete MLOps lifecycle using Azure Databricks.

The project focuses on the following objectives:

### 📂 Data Management

- Store the project datasets in a **Unity Catalog Volume**
- Access the datasets from the Databricks environment
- Prepare the data for machine learning

### 🤖 Machine Learning

- Develop a regression-based prediction system
- Train multiple machine learning algorithms
- Evaluate the models using appropriate regression metrics
- Compare model performance
- Select the best-performing model

### 📊 Experiment Tracking

- Use **MLflow** to track machine learning experiments
- Record model parameters
- Record evaluation metrics
- Track individual model runs
- Log trained model artifacts

### 🗂️ Model Management

- Log the selected model using MLflow
- Register the selected model in **Unity Catalog Model Registry**
- Manage the registered model and its version

### 🚀 Model Deployment

- Deploy the registered model using **Databricks Model Serving**
- Create a serving endpoint for the trained model
- Verify that the endpoint reaches a ready state

### 🔮 Model Inference

- Send prediction requests to the deployed serving endpoint
- Generate calories-burned predictions
- Compare the serving endpoint prediction with the local model prediction
- Validate the deployed model

### 🔄 Overall MLOps Objective

The overall objective is to demonstrate how a machine learning model can move through a practical MLOps lifecycle:

```text
Data
  ↓
Development
  ↓
Experimentation
  ↓
Model Selection
  ↓
Model Logging
  ↓
Model Registration
  ↓
Model Serving
  ↓
Inference
```

# 📊 Dataset

The project uses two datasets containing exercise and physiological information required for predicting calories burned.

### Dataset Files

```text
02-DATASET/
│
├── calories.csv
├── exercise.csv
└── README.md
```

### 📄 `exercise.csv`

The exercise dataset contains the input features used by the machine learning models.

| Feature | Description |
|---|---|
| `Gender` | Gender of the individual |
| `Age` | Age of the individual |
| `Height` | Height of the individual |
| `Weight` | Weight of the individual |
| `Duration` | Duration of the exercise session |
| `Heart_Rate` | Heart rate during exercise |
| `Body_Temp` | Body temperature during exercise |

### 📄 `calories.csv`

The calories dataset contains the target variable representing the calories burned during the exercise session.

| Column | Description |
|---|---|
| `User_ID` | Identifier of the individual |
| `Calories` | Number of calories burned |

The two datasets are combined using the common user identifier to create the final dataset used for machine learning.

### 🎯 Target Variable

The target variable for the prediction task is:

```text
Calories
```

The machine learning models learn the relationship between the exercise and physiological features and the corresponding calories burned.

### 📦 Data Storage

The raw datasets are stored in a **Unity Catalog Volume** within Azure Databricks.

```text
Unity Catalog
      │
      ▼
Catalog
      │
      ▼
Schema
      │
      ▼
Volume
      │
      ├── calories.csv
      │
      └── exercise.csv
```

The Databricks Notebook accesses the datasets from the Volume and loads them for further processing and model development.

# 🔄 MLOps Workflow

The project follows a structured MLOps workflow that takes the machine learning solution from raw data ingestion to a deployed model capable of serving predictions.

The workflow is divided into the following phases:

```text
01. Data Ingestion
        ↓
02. Development
        ↓
03. MLflow Experiment Tracking
        ↓
04. Model Logging
        ↓
05. Model Registry
        ↓
06. Model Serving
        ↓
07. Inference
```

Each phase represents a specific stage in the machine learning lifecycle implemented in this project.

---

## 1. 📥 Data Ingestion

The raw `calories.csv` and `exercise.csv` datasets are stored in a **Unity Catalog Volume** within Azure Databricks.

The Databricks Notebook accesses the datasets from the Volume and loads them for further processing.

```text
Raw CSV Files
      ↓
Unity Catalog Volume
      ↓
Databricks Notebook
      ↓
DataFrames
```

This stage establishes the data source used by the machine learning workflow.

📁 Related files:

[`04-Data-Ingestion/`](04-Data-Ingestion/)

---

## 2. 📒 Development

The machine learning workflow is implemented in a **Databricks Notebook**.

The notebook performs the core development activities required to prepare the data and build the prediction models.

The development stage includes:

- Loading the datasets
- Combining the required data
- Data preparation
- Exploratory analysis
- Feature preparation
- Train-test data preparation
- Model training
- Model evaluation

📁 Related files:

[`05-Development/`](05-Development/)

[`03-NOTEBOOKS/`](03-NOTEBOOKS/)

---

## 3. 📈 MLflow Experiment Tracking

Multiple regression models are trained and tracked using **MLflow**.

The models evaluated during experimentation include:

- Linear Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- XGBoost

Each model is tracked as an MLflow run along with its relevant parameters and evaluation metrics.

The primary evaluation metrics used are:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **R² — Coefficient of Determination**

The tracked runs allow the performance of the different models to be compared and the best-performing model to be selected.

📁 Related files:

[`06-MLflow-Experiment/`](06-MLflow-Experiment/)

---

## 4. 📦 Model Logging

After evaluating the different models, **XGBoost** is selected as the best-performing model.

The selected XGBoost model is logged using **MLflow**.

The logged model contains the model information and artifacts required for subsequent model management and deployment.

```text
Trained XGBoost Model
        ↓
MLflow
        ↓
Logged Model
        ↓
Model Artifacts
```

📁 Related files:

[`07-Model-Logging/`](07-Model-Logging/)

---

## 5. 🗂️ Model Registry

The logged XGBoost model is registered in the **Unity Catalog Model Registry**.

The registered model is:

```text
databricks00.default.calories_burnt_predictor
```

The Model Registry provides a managed location for the selected model and its version before deployment.

```text
MLflow Logged Model
        ↓
Unity Catalog Model Registry
        ↓
calories_burnt_predictor
        ↓
Model Version
```

📁 Related files:

[`08-Model-Registry/`](08-Model-Registry/)

---

## 6. 🚀 Model Serving

The registered model is deployed using **Databricks Model Serving**.

A serving endpoint is created for the registered model and configured to serve the selected model version.

The serving endpoint used in the project is:

```text
calories-burnt-predictor
```

The endpoint is verified until it reaches a ready state and can accept inference requests.

```text
Registered Model
        ↓
Model Version
        ↓
Databricks Model Serving
        ↓
calories-burnt-predictor
        ↓
Ready Endpoint
```

📁 Related files:

[`09-Model-Serving/`](09-Model-Serving/)

---

## 7. 🔮 Inference

Once the serving endpoint is ready, a sample input containing the required model features is sent to the endpoint.

The endpoint processes the input using the deployed XGBoost model and returns the predicted number of calories burned.

The deployed prediction is then compared with the prediction generated by the local model.

The validation produced:

```text
Local Model Prediction     : 233.57025
Serving Endpoint Prediction: 233.570426...
Difference                  : ~0.00017
```

The very small difference indicates that the deployed serving endpoint produced a prediction consistent with the local model.

📁 Related files:

[`10-Inference/`](10-Inference/)

# 📈 Model Development & Experimentation

The model development phase focuses on training multiple regression algorithms and evaluating their ability to predict calories burned.

The experiments are performed in the **Azure Databricks Notebook**, with **MLflow** used to track the individual model runs and their evaluation results.

---

## 🤖 Models Trained

Five regression algorithms were trained and evaluated:

| Model | Description |
|---|---|
| **Linear Regression** | Baseline linear regression model |
| **Decision Tree** | Tree-based regression model |
| **Random Forest** | Ensemble of multiple decision trees |
| **Gradient Boosting** | Sequential boosting-based regression model |
| **XGBoost** | Gradient boosting model optimized for performance |

Each model is trained using the prepared dataset and evaluated using the same evaluation metrics.

---

## 📊 Model Evaluation

The models are evaluated using three regression metrics:

### MAE — Mean Absolute Error

MAE measures the average absolute difference between the actual and predicted calorie values.

```text
MAE = Average(|Actual - Predicted|)
```

A lower MAE indicates better prediction accuracy.

### RMSE — Root Mean Squared Error

RMSE measures the square root of the average squared prediction error.

```text
RMSE = √(Average((Actual - Predicted)²))
```

A lower RMSE indicates better model performance and gives greater weight to larger prediction errors.

### R² — Coefficient of Determination

R² measures how well the model explains the variation in the target variable.

A value closer to `1` indicates a better fit to the evaluated data.

---

## 🧪 MLflow Experiment Tracking

Each model training process is recorded as an MLflow run.

The MLflow experiment stores information such as:

- Model parameters
- MAE
- RMSE
- R²
- Model artifacts
- Run information

This allows all trained models to be evaluated and compared within the same experiment.

```text
                 MLflow Experiment
                        │
       ┌────────────────┼────────────────┐
       │                │                │
       ▼                ▼                ▼
   Parameters        Metrics         Artifacts
       │                │                │
       └────────────────┼────────────────┘
                        ▼
                 Model Comparison
```

---

## 🏆 Model Selection

After evaluating all five models, **XGBoost** was selected as the final model for the MLOps pipeline.

The selected model achieved the following evaluation results:

| Metric | XGBoost |
|---|---:|
| **MAE** | **1.1589** |
| **RMSE** | **1.6728** |
| **R²** | **0.9993** |

The XGBoost model was therefore selected for the subsequent **model logging, model registration, and model serving** stages.

---

## 🔄 Experimentation Flow

```text
Prepared Dataset
       │
       ▼
┌──────────────────────────────────────┐
│          Model Training              │
├──────────────────────────────────────┤
│ Linear Regression                    │
│ Decision Tree                        │
│ Random Forest                        │
│ Gradient Boosting                    │
│ XGBoost                              │
└──────────────────┬───────────────────┘
                   │
                   ▼
           MLflow Experiment
              Tracking
                   │
                   ▼
           Metric Comparison
                   │
                   ▼
          Select Best Model
                   │
                   ▼
               XGBoost
```

---

## 📁 Project Evidence

The screenshots related to model experimentation, MLflow runs, and model comparison are available in:

[`06-MLflow-Experiment/`](06-MLflow-Experiment/)

The model development and training implementation is available in:

[`03-NOTEBOOKS/`](03-NOTEBOOKS/)


# 📦 Model Logging

After comparing the trained regression models, **XGBoost** was selected as the best-performing model and was used for the remaining MLOps lifecycle.

The selected model was logged using **MLflow**, creating a managed MLflow model artifact that can be used for subsequent model registration and deployment.

---

## 🎯 Why Model Logging?

Model logging captures the trained model together with the information required to load and use it later.

Instead of keeping the trained model only in the notebook session, MLflow stores the model as a persistent artifact associated with the corresponding experiment run.

This allows the model to be retrieved independently of the training process.

---

## 🔄 Model Logging Workflow

```text
Best Performing Model
        │
        ▼
     XGBoost
        │
        ▼
   MLflow Run
        │
        ▼
   Log Model
        │
        ▼
 Logged Model Artifact
        │
        ▼
Ready for Model Registration
```

---

## 🧪 MLflow Logged Model

The XGBoost model is logged under the MLflow run associated with the selected model.

The logged model contains the model artifact and supporting MLflow model information required for loading and serving the model.

The MLflow logged model includes artifacts such as:

```text
model/
├── MLmodel
├── model.pkl
├── conda.yaml
├── python_env.yaml
├── requirements.txt
├── input_example.json
└── serving_input_example.json
```

These artifacts allow the model to be reproduced and loaded through MLflow without requiring the original training session.

---

## 📊 Model Signature

During model logging, MLflow captures the model input schema.

The expected input features are:

```text
Gender
Age
Height
Weight
Duration
Heart_Rate
Body_Temp
```

The model signature helps ensure that inference requests contain the expected input columns and compatible data types.

---

## 🔗 MLflow Model

The logged XGBoost model is identified by its MLflow model ID and associated training run.

This logged model is subsequently used as the source for registration in the **Unity Catalog Model Registry**.

```text
MLflow Experiment
       │
       ▼
Selected XGBoost Run
       │
       ▼
MLflow Logged Model
       │
       ▼
Unity Catalog Model Registry
```

---

## 📸 Project Evidence

The screenshots demonstrating the XGBoost MLflow run, model metrics, and logged model artifacts are available in:

[`07-Model-Logging/`](07-Model-Logging/)

# 🗂️ Model Registry

After logging the selected XGBoost model with MLflow, the model is registered in the **Unity Catalog Model Registry**.

Model registration provides a centralized location to manage the trained model and its versions before deployment.

---

## 🎯 Registered Model

The selected model was registered with the following three-level Unity Catalog model name:

```text
databricks00.default.calories_burnt_predictor
```

Where:

```text
Catalog
   ↓
databricks00

Schema
   ↓
default

Registered Model
   ↓
calories_burnt_predictor
```

---

## 🔄 Model Registration Workflow

```text
MLflow Logged Model
        │
        ▼
Unity Catalog
        │
        ▼
Catalog: databricks00
        │
        ▼
Schema: default
        │
        ▼
calories_burnt_predictor
        │
        ▼
Registered Model Version
```

---

## 📦 Model Version

The registered model is maintained as a versioned model in Unity Catalog.

The model version represents a specific trained model artifact that can be selected for deployment.

This creates a clear separation between:

- The model produced during experimentation
- The registered model
- The specific model version used for deployment

---

## 🔗 MLflow to Unity Catalog

The model lifecycle at this stage can be represented as:

```text
MLflow Experiment
        │
        ▼
XGBoost Run
        │
        ▼
Logged Model
        │
        ▼
Unity Catalog Model Registry
        │
        ▼
calories_burnt_predictor
        │
        ▼
Model Version
```

The registered model is then used as the source model for the **Databricks Model Serving** stage.

---

## 📸 Project Evidence

Screenshots demonstrating the registered model and its details are available in:

[`08-Model-Registry/`](08-Model-Registry/)


