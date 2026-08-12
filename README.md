# 🔥 Calories Burnt Prediction – End-to-End MLOps on Databricks

> An end-to-end machine learning and MLOps project for predicting calories burned during physical activity using **Azure Databricks, Unity Catalog, MLflow, and Databricks Model Serving**.

---

## 📑 Project Index

- [🔥 Calories Burnt Prediction – End-to-End MLOps on Databricks](#-calories-burnt-prediction--end-to-end-mlops-on-databricks)
  - [📑 Project Index](#-project-index)
- [📌 Project Overview](#-project-overview)
- [🎯 Objectives](#-objectives)
- [🏗️ Architecture](#️-architecture)
    - [End-to-End Flow](#end-to-end-flow)
- [🛠️ Technologies Used](#️-technologies-used)
- [☁️ Azure \& Databricks Services Used](#️-azure--databricks-services-used)
- [📊 Dataset](#-dataset)
    - [Input Features](#input-features)
    - [Target Variable](#target-variable)
- [🔄 MLOps Workflow](#-mlops-workflow)
  - [1. Data Ingestion](#1-data-ingestion)
  - [2. Development](#2-development)
  - [3. MLflow Experiment Tracking](#3-mlflow-experiment-tracking)
  - [4. Model Logging](#4-model-logging)
  - [5. Model Registry](#5-model-registry)
    - [Registered Model](#registered-model)
  - [6. Model Serving](#6-model-serving)
    - [Serving Endpoint](#serving-endpoint)
  - [7. Inference](#7-inference)
    - [Prediction Validation](#prediction-validation)
- [📈 Model Results](#-model-results)
- [🚀 Deployment](#-deployment)
- [📂 Repository Structure](#-repository-structure)
- [▶️ How to Run](#️-how-to-run)
  - [Prerequisites](#prerequisites)
  - [Step 1 — Prepare the Databricks Environment](#step-1--prepare-the-databricks-environment)
  - [Step 2 — Upload the Dataset](#step-2--upload-the-dataset)
  - [Step 3 — Import the Notebook](#step-3--import-the-notebook)
  - [Step 4 — Configure the Dataset Path](#step-4--configure-the-dataset-path)
  - [Step 5 — Run the Notebook](#step-5--run-the-notebook)
  - [Step 6 — Deploy the Model](#step-6--deploy-the-model)
  - [Step 7 — Test the Endpoint](#step-7--test-the-endpoint)
- [🔮 Future Improvements](#-future-improvements)
- [🧠 Skills Demonstrated](#-skills-demonstrated)
- [📜 License](#-license)
- [👤 Author](#-author)

---

# 📌 Project Overview

This project demonstrates an **end-to-end Machine Learning and MLOps workflow** for predicting calories burned during physical activity using the Azure Databricks ecosystem.

The project covers the complete lifecycle of a machine learning model, starting with dataset storage and ingestion, followed by data preparation, model development, experiment tracking, model logging, model registration, model serving, and real-time inference.

The machine learning solution uses exercise and physiological attributes such as gender, age, height, weight, exercise duration, heart rate, and body temperature to predict calories burned.

Multiple regression models are trained and evaluated. **MLflow** is used to track experiments, parameters, metrics, runs, and model artifacts. After model comparison, **XGBoost** is selected as the final model.

The selected model is logged using MLflow, registered in the **Unity Catalog Model Registry**, and deployed through **Databricks Model Serving**.

The deployed serving endpoint is then tested with sample input data. The serving prediction is compared with the local model prediction to validate that the deployed model produces a consistent result.

---

# 🎯 Objectives

The main objectives of the project are:

- Store and access datasets using **Unity Catalog Volumes**
- Perform data preparation and development in **Azure Databricks**
- Train multiple regression models
- Evaluate models using MAE, RMSE, and R²
- Track experiments using **MLflow**
- Compare model performance and select the best model
- Log the selected model using MLflow
- Register the model in **Unity Catalog Model Registry**
- Deploy the registered model using **Databricks Model Serving**
- Perform real-time inference
- Validate the serving endpoint against the local model

---

# 🏗️ Architecture

The project is organized around the following core components:

- **Unity Catalog Volume** – stores the raw datasets
- **Databricks Notebook** – performs data preparation, experimentation, training, and evaluation
- **MLflow** – tracks experiments and logs models
- **Unity Catalog Model Registry** – registers and versions the selected model
- **Databricks Model Serving** – deploys the registered model
- **Serving Endpoint** – provides real-time predictions

### End-to-End Flow

```text
Raw Datasets
     ↓
Unity Catalog Volume
     ↓
Databricks Notebook
     ↓
Data Preparation
     ↓
Model Training
     ↓
MLflow Experiment Tracking
     ↓
Model Comparison
     ↓
Best Model – XGBoost
     ↓
MLflow Model Logging
     ↓
Unity Catalog Model Registry
     ↓
Databricks Model Serving
     ↓
Inference Endpoint
     ↓
Prediction
```

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| **Microsoft Azure** | Cloud platform |
| **Azure Databricks** | Machine learning development and execution |
| **Python** | Data processing and machine learning |
| **Pandas** | Dataset loading and manipulation |
| **Scikit-learn** | Regression models and evaluation |
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

---

# 📊 Dataset

The project uses two datasets:

- `calories.csv`
- `exercise.csv`

The datasets contain exercise and physiological information used to predict calories burned.

### Input Features

| Feature | Description |
|---|---|
| `Gender` | Gender of the individual |
| `Age` | Age of the individual |
| `Height` | Height of the individual |
| `Weight` | Weight of the individual |
| `Duration` | Exercise duration |
| `Heart_Rate` | Heart rate during exercise |
| `Body_Temp` | Body temperature during exercise |

### Target Variable

```text
Calories
```

The two datasets are combined using the common user identifier to create the dataset used for model development.

The raw files are stored in a **Unity Catalog Volume** and accessed from the Databricks Notebook.

---

# 🔄 MLOps Workflow

## 1. Data Ingestion

The raw CSV datasets are stored in a Unity Catalog Volume within Azure Databricks.

The Databricks Notebook loads the datasets from the Volume and prepares them for further processing.

```text
calories.csv
     │
exercise.csv
     │
     ▼
Unity Catalog Volume
     │
     ▼
Databricks Notebook
     │
     ▼
DataFrames
```

📁 Related files:

[`04-Data-Ingestion/`](04-Data-Ingestion/)

---

## 2. Development

The Databricks Notebook is used as the main development environment.

The development stage includes:

- Loading the datasets
- Combining the required data
- Data preparation
- Exploratory analysis
- Feature preparation
- Model training
- Model evaluation

📁 Related files:

[`03-NOTEBOOKS/`](03-NOTEBOOKS/)

[`05-Development/`](05-Development/)

---

## 3. MLflow Experiment Tracking

Five regression models are trained and evaluated:

1. Linear Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

Each model is tracked as an MLflow run.

The main evaluation metrics are:

- **MAE — Mean Absolute Error**
- **RMSE — Root Mean Squared Error**
- **R² — Coefficient of Determination**

MLflow records the relevant model parameters, metrics, run information, and artifacts.

📁 Related files:

[`06-MLflow-Experiment/`](06-MLflow-Experiment/)

---

## 4. Model Logging

After comparing the trained models, **XGBoost** is selected as the final model.

The selected XGBoost model is logged using MLflow.

The logged model contains the model artifact and supporting MLflow information required for subsequent registration and deployment.

```text
Best Model
    ↓
XGBoost
    ↓
MLflow Model Logging
    ↓
Logged Model
    ↓
Model Artifacts
```

📁 Related files:

[`07-Model-Logging/`](07-Model-Logging/)

---

## 5. Model Registry

The logged XGBoost model is registered in the Unity Catalog Model Registry.

### Registered Model

```text
databricks00.default.calories_burnt_predictor
```

The registered model provides a managed location for the selected model and its version before deployment.

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

## 6. Model Serving

The registered XGBoost model is deployed using Databricks Model Serving.

### Serving Endpoint

```text
calories-burnt-predictor
```

The endpoint is configured to serve the registered model version and is verified until it reaches the **Ready** state.

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

## 7. Inference

Once the serving endpoint is ready, a sample input is sent to the deployed model.

The endpoint returns the predicted number of calories burned.

The serving prediction is then compared with the local model prediction.

### Prediction Validation

```text
Local Model Prediction      : 233.57025
Serving Endpoint Prediction : 233.570426...
Difference                   : ~0.00017
```

The very small difference demonstrates that the deployed endpoint produces a prediction consistent with the local model.

📁 Related files:

[`10-Inference/`](10-Inference/)

---

# 📈 Model Results

The models were evaluated using MAE, RMSE, and R².

The final XGBoost evaluation results were:

| Metric | XGBoost |
|---|---:|
| **MAE** | **1.1589** |
| **RMSE** | **1.6728** |
| **R²** | **0.9993** |

Based on the experimentation results, **XGBoost** was selected as the final model for logging, registration, and deployment.

> The reported metrics represent the evaluation performed in this project and should not be interpreted as a guarantee of performance on unseen production data.

---

# 🚀 Deployment

The final model deployment flow is:

```text
XGBoost Model
     ↓
MLflow Logged Model
     ↓
Unity Catalog Model Registry
     ↓
Registered Model Version
     ↓
Databricks Model Serving
     ↓
calories-burnt-predictor
     ↓
Inference Request
     ↓
Prediction Response
```

The serving endpoint was successfully brought to the **Ready** state and tested using a sample prediction request.

The deployment endpoint URL is not included in this repository because it is specific to the Databricks workspace environment.

---

# 📂 Repository Structure

```text
Calories-Burnt-Prediction-MLOps/
│
├── 01-ARCHITECTURE/
│   └── calories-burnt-mlops-architecture.png
│
├── 02-DATASET/
│   ├── calories.csv
│   └── exercise.csv
│
├── 03-NOTEBOOKS/
│   └── Calories_Burnt_MLOps.ipynb
│
├── 04-Data-Ingestion/
│   └── Project screenshots
│
├── 05-Development/
│   └── Project screenshots
│
├── 06-MLflow-Experiment/
│   └── Project screenshots
│
├── 07-Model-Logging/
│   └── Project screenshots
│
├── 08-Model-Registry/
│   └── Project screenshots
│
├── 09-Model-Serving/
│   └── Project screenshots
│
├── 10-Inference/
│   └── Project screenshots
│
├── .gitignore
├── LICENSE
└── README.md
```

---

# ▶️ How to Run

## Prerequisites

- Microsoft Azure account
- Azure Databricks workspace
- Unity Catalog enabled
- Databricks compute resource
- GitHub account

## Step 1 — Prepare the Databricks Environment

Create or access an Azure Databricks workspace with Unity Catalog enabled.

Create the required catalog, schema, and Unity Catalog Volume.

## Step 2 — Upload the Dataset

Upload:

```text
calories.csv
exercise.csv
```

to the Unity Catalog Volume.

## Step 3 — Import the Notebook

Import the notebook from:

```text
03-NOTEBOOKS/Calories_Burnt_MLOps.ipynb
```

into the Databricks workspace.

## Step 4 — Configure the Dataset Path

Update the Unity Catalog Volume path in the notebook if your catalog, schema, or Volume names differ from the environment used in this project.

## Step 5 — Run the Notebook

Execute the notebook sequentially to perform:

```text
Data Ingestion
     ↓
Data Preparation
     ↓
Model Training
     ↓
Model Evaluation
     ↓
MLflow Tracking
     ↓
Model Logging
     ↓
Model Registration
```

## Step 6 — Deploy the Model

After registering the selected model, create a Databricks Model Serving endpoint using the registered model version.

## Step 7 — Test the Endpoint

Send a prediction request containing the required input features and verify the returned prediction.

---

# 🔮 Future Improvements

The current implementation focuses on the core MLOps lifecycle from data ingestion through model serving and inference.

Possible future improvements include:

- Automated model retraining
- Automated deployment pipelines
- CI/CD integration
- Data quality validation
- Automated model evaluation gates
- Model performance monitoring
- Data and model drift detection
- Scheduled batch inference
- Integration with a frontend or application

These improvements are outside the scope of the current implementation.

---

# 🧠 Skills Demonstrated

This project demonstrates practical experience with:

- Azure Databricks
- Unity Catalog
- Unity Catalog Volumes
- Python
- Pandas
- Scikit-learn
- XGBoost
- Regression model development
- Model evaluation
- MLflow Experiment Tracking
- MLflow Model Logging
- Unity Catalog Model Registry
- Model version management
- Databricks Model Serving
- REST-based inference
- Model deployment validation
- GitHub project organization

---

# 📜 License

This project is licensed under the terms specified in the [`LICENSE`](LICENSE) file.

---

# 👤 Author

**Sohail Akhter**

Computer Science | Machine Learning & Data Engineering
