# Databricks notebook source
# MAGIC %md
# MAGIC # Calories Burnt Prediction System
# MAGIC
# MAGIC ## Phase 1 - Data Engineering
# MAGIC
# MAGIC ### Step 1 - Import Required Libraries
# MAGIC
# MAGIC In this section, we import the Python libraries required for data manipulation and numerical computations. These libraries will be used throughout the project.

# COMMAND ----------

import pandas as pd
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 2: Read the Datasets
# MAGIC
# MAGIC The raw datasets are stored inside a Unity Catalog Volume.
# MAGIC
# MAGIC In this step, we load both datasets into pandas DataFrames.

# COMMAND ----------

exercise = pd.read_csv(
    "/Volumes/databricks00/default/calories_data/exercise(1).csv"
)

calories = pd.read_csv(
    "/Volumes/databricks00/default/calories_data/calories(1).csv"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 3: Display Sample Records
# MAGIC
# MAGIC Before performing any preprocessing, it is good practice to inspect a few records from each dataset.

# COMMAND ----------

exercise.head()

# COMMAND ----------

calories.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 4: Check Dataset Dimensions
# MAGIC
# MAGIC The shape of a dataset tells us the number of rows and columns available for analysis.

# COMMAND ----------

print(f"Exercise Dataset Shape : {exercise.shape}")
print(f"Calories Dataset Shape : {calories.shape}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 5: Inspect Dataset Information
# MAGIC
# MAGIC This step helps us verify:
# MAGIC - Column names
# MAGIC - Data types
# MAGIC - Missing values
# MAGIC - Memory usage

# COMMAND ----------

exercise.info()

print("=" * 80)

calories.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 6: Check for Missing Values
# MAGIC
# MAGIC Before merging the datasets, we verify whether any column contains missing (null) values.
# MAGIC
# MAGIC Missing values can negatively impact data quality and model performance.

# COMMAND ----------

print("Exercise Dataset")
print(exercise.isnull().sum())

print("\n" + "="*60 + "\n")

print("Calories Dataset")
print(calories.isnull().sum())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 7: Check for Duplicate Records
# MAGIC
# MAGIC Duplicate records may introduce bias into the machine learning model.
# MAGIC
# MAGIC We first check whether either dataset contains duplicate rows.

# COMMAND ----------

print(f"Duplicate rows in Exercise Dataset : {exercise.duplicated().sum()}")
print(f"Duplicate rows in Calories Dataset : {calories.duplicated().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 8: Check for Duplicate User IDs
# MAGIC
# MAGIC Each user should appear exactly once in both datasets.
# MAGIC
# MAGIC If duplicate User_ID values exist, merging may produce incorrect results.

# COMMAND ----------

print(f"Duplicate User_IDs in Exercise Dataset : {exercise['User_ID'].duplicated().sum()}")

print(f"Duplicate User_IDs in Calories Dataset : {calories['User_ID'].duplicated().sum()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 9: Merge the Datasets
# MAGIC
# MAGIC Both datasets share a common column named **User_ID**.
# MAGIC
# MAGIC We perform an inner join to combine demographic, exercise, and calorie information into a single dataset.

# COMMAND ----------

df = pd.merge(
    exercise,
    calories,
    on="User_ID",
    how="inner"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 10: Verify the Merge
# MAGIC
# MAGIC After merging, we verify that the merge completed successfully by checking the dataset dimensions.

# COMMAND ----------

print(df.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 11: get first 5 records
# MAGIC

# COMMAND ----------

df.head()


# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 12: Describe dataset

# COMMAND ----------

df.info()

# COMMAND ----------

# MAGIC %md
# MAGIC ## EXPLORATORY DATA ANALYSIS 

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 13: Analyze Numerical Features
# MAGIC
# MAGIC We begin the Exploratory Data Analysis by generating a statistical summary of the numerical features.
# MAGIC
# MAGIC This helps us understand the central tendency, spread, and range of the data.

# COMMAND ----------

df.describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 14: Analyze Gender Distribution
# MAGIC
# MAGIC Gender is a categorical feature in our dataset.
# MAGIC
# MAGIC We will examine the number of records belonging to each gender category.

# COMMAND ----------

df["Gender"].value_counts()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 15: Calculate Gender Distribution Percentage
# MAGIC
# MAGIC We calculate the percentage distribution of each gender category to understand whether the dataset is reasonably balanced.

# COMMAND ----------

df["Gender"].value_counts(normalize=True) * 100

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 16: Visualize Gender Distribution
# MAGIC
# MAGIC A bar chart provides a visual representation of the number of records in each gender category.

# COMMAND ----------

import matplotlib.pyplot as plt

df["Gender"].value_counts().plot(kind="bar")

plt.title("Gender Distribution")
plt.xlabel("Gender")
plt.ylabel("Number of Records")
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 17: Analyze the Target Variable
# MAGIC
# MAGIC The target variable for our regression problem is `Calories`.
# MAGIC
# MAGIC We examine its statistical properties to understand the range and distribution of the values we want the model to predict.

# COMMAND ----------

df["Calories"].describe()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 18: Visualize Calories Distribution
# MAGIC
# MAGIC A histogram helps us understand how the target variable is distributed across the dataset.

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.hist(df["Calories"], bins=30)

plt.title("Distribution of Calories Burnt")
plt.xlabel("Calories")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 19: Analyze Numerical Feature Distributions
# MAGIC
# MAGIC We visualize the distributions of the numerical input features.
# MAGIC
# MAGIC This can help us identify unusual distributions and potential extreme values that may require further investigation.

# COMMAND ----------

numerical_columns = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"
]

df[numerical_columns].hist(
    figsize=(14, 10),
    bins=30
)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 20: Calculate Feature Correlations
# MAGIC
# MAGIC Correlation helps us understand the strength and direction of linear relationships between numerical variables.
# MAGIC
# MAGIC We are particularly interested in how each feature is correlated with `Calories`.

# COMMAND ----------

correlation_matrix = df[numerical_columns + ["Calories"]].corr()

correlation_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 21: Visualize the Correlation Matrix
# MAGIC
# MAGIC A heatmap-style visualization makes it easier to identify strong and weak correlations between numerical variables.

# COMMAND ----------

plt.figure(figsize=(10, 7))

plt.imshow(correlation_matrix, cmap="coolwarm")

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Matrix")

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 22: Identify Features Correlated with Calories
# MAGIC
# MAGIC We sort the correlation values for `Calories` in descending order.
# MAGIC
# MAGIC This allows us to quickly identify which numerical features have the strongest linear relationship with the target variable.

# COMMAND ----------

calorie_correlation = (
    correlation_matrix["Calories"]
    .sort_values(ascending=False)
)

calorie_correlation

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 23: Analyze Exercise Duration vs Calories
# MAGIC
# MAGIC Exercise duration is an important candidate feature.
# MAGIC
# MAGIC We visualize its relationship with calories burnt to determine whether longer exercise sessions are associated with higher calorie expenditure.

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Duration"],
    df["Calories"],
    alpha=0.5
)

plt.title("Exercise Duration vs Calories Burnt")
plt.xlabel("Duration")
plt.ylabel("Calories")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 24: Analyze Heart Rate vs Calories
# MAGIC
# MAGIC We visualize the relationship between heart rate and calories burnt to determine whether higher heart rates are associated with greater calorie expenditure.

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Heart_Rate"],
    df["Calories"],
    alpha=0.5
)

plt.title("Heart Rate vs Calories Burnt")
plt.xlabel("Heart Rate")
plt.ylabel("Calories")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 25: Analyze Weight vs Calories
# MAGIC
# MAGIC We visualize the relationship between body weight and calories burnt to determine whether weight has a noticeable relationship with the target variable.

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.scatter(
    df["Weight"],
    df["Calories"],
    alpha=0.5
)

plt.title("Weight vs Calories Burnt")
plt.xlabel("Weight")
plt.ylabel("Calories")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 26: Detect Potential Outliers
# MAGIC
# MAGIC Boxplots can help us identify observations that are unusually far from the rest of the data.
# MAGIC
# MAGIC An outlier is not automatically an error. Any potential outlier will be investigated before deciding whether it should be removed.

# COMMAND ----------

plt.figure(figsize=(14, 8))

df[numerical_columns].boxplot()

plt.title("Boxplot of Numerical Features")
plt.xticks(rotation=45)

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 27: Summarize EDA Findings
# MAGIC
# MAGIC ### Step 27: EDA Findings
# MAGIC
# MAGIC The exploratory data analysis provides the following observations:
# MAGIC
# MAGIC - `Duration` shows a strong positive relationship with `Calories`.
# MAGIC - `Heart_Rate` shows a strong positive relationship with `Calories`.
# MAGIC - `Body_Temp` also shows a strong positive relationship with `Calories`.
# MAGIC - `Age` has a relatively weak relationship with `Calories`.
# MAGIC - `Height` and `Weight` show comparatively weaker linear relationships with the target.
# MAGIC - Potential outliers are visible in some numerical features, particularly `Height`, `Weight`, `Heart_Rate`, and `Body_Temp`.
# MAGIC - The presence of an outlier does not necessarily indicate an incorrect observation, so the observations will not be removed without further validation.
# MAGIC - `Gender` is a categorical feature and will require encoding before model training.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 28: Define Features and Target
# MAGIC
# MAGIC The objective of this project is to predict the number of calories burnt during exercise.
# MAGIC
# MAGIC `Calories` is therefore defined as the target variable.
# MAGIC
# MAGIC `User_ID` is an identifier and does not provide meaningful predictive information, so it will be excluded from the feature set.
# MAGIC
# MAGIC The remaining columns will be used as input features.

# COMMAND ----------

# Define the target variable
y = df["Calories"]

# Remove the target and identifier columns from the features
X = df.drop(columns=["Calories", "User_ID"])

print("Feature columns:")
print(X.columns.tolist())

print("\nTarget column:")
print(y.name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 29: Validate Feature and Target Dimensions
# MAGIC
# MAGIC We verify that the feature matrix and target variable contain the same number of observations before proceeding with model development.

# COMMAND ----------

print("Features shape :", X.shape)
print("Target shape   :", y.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 4: Data Preprocessing and Feature Engineering
# MAGIC
# MAGIC Machine learning algorithms require numerical input features.
# MAGIC
# MAGIC Our dataset contains one categorical feature, `Gender`, while the remaining features are numerical.
# MAGIC
# MAGIC In this phase, we will:
# MAGIC
# MAGIC - Encode the categorical feature.
# MAGIC - Separate numerical and categorical features.
# MAGIC - Build a preprocessing pipeline.
# MAGIC - Split the data into training and testing sets.
# MAGIC
# MAGIC The preprocessing steps will be fitted only on the training data to prevent data leakage.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 30: Identify Categorical and Numerical Features
# MAGIC
# MAGIC We explicitly identify the categorical and numerical columns so that each type of feature can receive the appropriate preprocessing.

# COMMAND ----------

categorical_features = ["Gender"]

numerical_features = [
    "Age",
    "Height",
    "Weight",
    "Duration",
    "Heart_Rate",
    "Body_Temp"
]

print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 31: Inspect the Gender Values
# MAGIC
# MAGIC Before encoding the categorical feature, we check the unique values present in the column.
# MAGIC
# MAGIC This ensures that we understand the categories that the preprocessing pipeline needs to handle.

# COMMAND ----------

X["Gender"].unique()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 32: Split the Dataset into Training and Testing Sets
# MAGIC
# MAGIC We split the data before fitting the preprocessing transformations.
# MAGIC
# MAGIC The training set will be used to learn the model and preprocessing parameters, while the test set will be reserved for final evaluation.

# COMMAND ----------

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 33: Create the Preprocessing Pipeline
# MAGIC
# MAGIC We use `ColumnTransformer` to apply different transformations to different types of features.
# MAGIC
# MAGIC `OneHotEncoder` converts the categorical `Gender` column into numerical columns.
# MAGIC
# MAGIC Numerical features are passed through unchanged because tree-based models such as Random Forest and XGBoost do not require feature scaling.

# COMMAND ----------

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 34: Fit the Preprocessing Pipeline on Training Data
# MAGIC
# MAGIC The preprocessing pipeline is fitted using only the training data.
# MAGIC
# MAGIC This is important because information from the test set must not influence the preprocessing process.
# MAGIC
# MAGIC This prevents data leakage.

# COMMAND ----------

preprocessor.fit(X_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 35: Transform Training and Testing Data
# MAGIC
# MAGIC After fitting the preprocessing pipeline on the training data, we transform both the training and testing datasets using the same fitted transformer.

# COMMAND ----------

X_train_processed = preprocessor.transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print("Processed X_train shape:", X_train_processed.shape)
print("Processed X_test shape :", X_test_processed.shape)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 36: Verify the Processed Features
# MAGIC
# MAGIC We inspect the feature names generated by the preprocessing pipeline.
# MAGIC
# MAGIC This allows us to verify exactly what will be provided to the machine learning model.

# COMMAND ----------

feature_names = preprocessor.get_feature_names_out()

print(feature_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 5: Model Development
# MAGIC
# MAGIC In this phase, we build the first machine learning model for predicting calories burnt.
# MAGIC
# MAGIC We will combine the preprocessing stage and the XGBoost regression model into a single pipeline.
# MAGIC
# MAGIC The pipeline will ensure that the same preprocessing logic used during training is automatically applied when making predictions on new data.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 37: Import the XGBoost Regressor
# MAGIC
# MAGIC XGBoost is a gradient boosting algorithm that can be used for regression problems.
# MAGIC
# MAGIC We will use `XGBRegressor` because our target variable, `Calories`, is continuous.

# COMMAND ----------

from xgboost import XGBRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 38: Create the XGBoost Model
# MAGIC
# MAGIC We create our first XGBoost regression model using a simple set of hyperparameters.
# MAGIC
# MAGIC These values will serve as our baseline configuration. Later, we will experiment with different hyperparameters and use MLflow to compare the results.

# COMMAND ----------

xgb_model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 39: Create the Complete ML Pipeline
# MAGIC
# MAGIC We combine the preprocessing component and the XGBoost model into a single scikit-learn pipeline.
# MAGIC
# MAGIC This ensures that preprocessing and prediction are treated as one complete workflow.

# COMMAND ----------

from sklearn.pipeline import Pipeline

model_pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", xgb_model)
    ]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 40: Train the Model
# MAGIC
# MAGIC We train the complete pipeline using the training dataset.
# MAGIC
# MAGIC The preprocessing transformation is fitted as part of the pipeline using only the training data.

# COMMAND ----------

model_pipeline.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 41: Generate Predictions
# MAGIC
# MAGIC We use the trained pipeline to generate predictions for the test dataset.
# MAGIC
# MAGIC The pipeline automatically performs the required preprocessing before passing the data to XGBoost.

# COMMAND ----------

y_pred = model_pipeline.predict(X_test)

print("First 10 predictions:")
print(y_pred[:10])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 42: Evaluate the Model
# MAGIC
# MAGIC We evaluate the model using three regression metrics:
# MAGIC
# MAGIC - Mean Absolute Error (MAE)
# MAGIC - Root Mean Squared Error (RMSE)
# MAGIC - R² Score
# MAGIC
# MAGIC These metrics provide different perspectives on model performance.

# COMMAND ----------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R²   : {r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 43: Visualize Actual vs Predicted Values
# MAGIC
# MAGIC A scatter plot allows us to visually compare the actual calorie values with the values predicted by the model.
# MAGIC
# MAGIC A model with strong predictive performance should produce predictions that are generally close to the actual values.

# COMMAND ----------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.5
)

plt.xlabel("Actual Calories")
plt.ylabel("Predicted Calories")
plt.title("Actual vs Predicted Calories")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 44: Analyze Prediction Errors
# MAGIC
# MAGIC We calculate the prediction error for each test observation.
# MAGIC
# MAGIC The residual is the difference between the actual value and the predicted value.

# COMMAND ----------

residuals = y_test - y_pred

print("Mean residual:", residuals.mean())
print("Minimum residual:", residuals.min())
print("Maximum residual:", residuals.max())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 45: Visualize Prediction Errors
# MAGIC
# MAGIC We visualize the distribution of residuals to understand how the model's prediction errors are distributed.

# COMMAND ----------

plt.figure(figsize=(8, 5))

plt.hist(residuals, bins=30)

plt.title("Distribution of Prediction Errors")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")

plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Phase 5: Model Comparison
# MAGIC
# MAGIC In this phase, we train multiple regression algorithms using the same training and testing datasets.
# MAGIC
# MAGIC The purpose is to determine which algorithm provides the best predictive performance for the Calories Burnt prediction problem.
# MAGIC
# MAGIC All models will use the same preprocessing pipeline and the same train-test split so that their performance can be compared fairly.
# MAGIC
# MAGIC The models we will evaluate are:
# MAGIC
# MAGIC 1. Linear Regression
# MAGIC 2. Decision Tree Regressor
# MAGIC 3. Random Forest Regressor
# MAGIC 4. Gradient Boosting Regressor
# MAGIC 5. XGBoost Regressor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 46: Import the Regression Algorithms
# MAGIC
# MAGIC We import the five regression algorithms that will be evaluated.

# COMMAND ----------

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 47: Create the Model Definitions
# MAGIC
# MAGIC We create the five models using reasonable baseline hyperparameters.
# MAGIC
# MAGIC At this stage, we are not performing hyperparameter tuning. The goal is to compare the baseline performance of different algorithms.

# COMMAND ----------

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        max_depth=10,
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    ),

    "XGBoost": XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42
    )
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 48: Train and Evaluate All Models
# MAGIC
# MAGIC Each model will be placed inside the same preprocessing pipeline.
# MAGIC
# MAGIC For every model, we will:
# MAGIC
# MAGIC - Apply the preprocessing pipeline.
# MAGIC - Train the model on the training data.
# MAGIC - Generate predictions on the test data.
# MAGIC - Calculate MAE.
# MAGIC - Calculate RMSE.
# MAGIC - Calculate R².
# MAGIC - Store the results for comparison.

# COMMAND ----------

from sklearn.base import clone

results = []
trained_models = {}

for model_name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", model)
        ]
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    predictions = pipeline.predict(X_test)

    # Metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    # Store results
    results.append({
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    # Store trained pipeline
    trained_models[model_name] = pipeline

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 49: Create the Model Comparison Table
# MAGIC
# MAGIC We convert the evaluation results into a DataFrame so that the performance of all models can be compared easily.

# COMMAND ----------

results_df = pd.DataFrame(results)

results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 50: Sort Models by MAE
# MAGIC
# MAGIC MAE represents the average absolute difference between the actual and predicted calorie values.
# MAGIC
# MAGIC Since lower MAE indicates smaller prediction errors, we sort the models from best to worst based on MAE.

# COMMAND ----------

results_df = results_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 51: Sort Models by RMSE
# MAGIC
# MAGIC RMSE gives greater weight to larger prediction errors.
# MAGIC
# MAGIC We use it as a secondary metric when comparing the models.

# COMMAND ----------

results_df.sort_values(
    by="RMSE",
    ascending=True
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 52: Compare Models Using R²
# MAGIC
# MAGIC R² measures how much of the variation in the target variable is explained by the model.
# MAGIC
# MAGIC Higher R² indicates better performance.

# COMMAND ----------

results_df.sort_values(
    by="R2",
    ascending=False
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 53: Visualize Model Performance
# MAGIC
# MAGIC We visualize the MAE of the five models to make the performance comparison easier to interpret.
# MAGIC
# MAGIC Lower MAE indicates better performance.

# COMMAND ----------

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["MAE"]
)

plt.title("Model Comparison - MAE")
plt.xlabel("Model")
plt.ylabel("Mean Absolute Error")

plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 54: Select the Best Model
# MAGIC
# MAGIC The model with the lowest MAE is selected as the initial candidate for deployment.
# MAGIC
# MAGIC RMSE and R² will also be considered to ensure that the model does not have unusually large errors or poor overall explanatory performance.

# COMMAND ----------

best_model_name = results_df.loc[0, "Model"]

best_model = trained_models[best_model_name]

print("Best Model:", best_model_name)

print("\nPerformance:")
print(results_df.iloc[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 55: Display the Final Model Comparison
# MAGIC
# MAGIC The final comparison table provides a consolidated view of all models evaluated during this experiment.

# COMMAND ----------

results_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 56: Interpret Model Comparison
# MAGIC
# MAGIC Five regression models were trained and evaluated using the same training and testing datasets.
# MAGIC
# MAGIC XGBoost achieved the best overall performance:
# MAGIC
# MAGIC - MAE: 1.158857
# MAGIC - RMSE: 1.672760
# MAGIC - R²: 0.999307
# MAGIC
# MAGIC XGBoost achieved the lowest MAE and RMSE and the highest R² among all evaluated models.
# MAGIC
# MAGIC Therefore, XGBoost is selected as the candidate model for the next stage of the MLOps workflow.
# MAGIC
# MAGIC The remaining models will not be discarded. Their evaluation results will be tracked as part of the experiment history.

# COMMAND ----------

best_model_name = results_df.loc[0, "Model"]
best_model = trained_models[best_model_name]

print("Selected Model:", best_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 57: Import MLflow
# MAGIC
# MAGIC MLflow is used to track machine learning experiments.
# MAGIC
# MAGIC In this phase, we will use MLflow to record the parameters, metrics, and trained models for each of our five experiments.

# COMMAND ----------

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 58: Check MLflow Version
# MAGIC
# MAGIC We check the installed MLflow version to confirm that MLflow is available in our Databricks compute environment.

# COMMAND ----------

print("MLflow version:", mlflow.__version__)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 59: Identify the Current Notebook Path
# MAGIC
# MAGIC MLflow experiments in Databricks can be organized using workspace paths.
# MAGIC
# MAGIC We retrieve the current notebook path so that our MLflow experiment can be created in the same project area.

# COMMAND ----------

notebook_path = (
    dbutils.notebook.entry_point
    .getDbutils()
    .notebook()
    .getContext()
    .notebookPath()
    .get()
)

print(notebook_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 60: Define the MLflow Experiment Path
# MAGIC
# MAGIC We create a dedicated MLflow Experiment inside our `Calories_Burnt_MLOps` project.
# MAGIC
# MAGIC The experiment will contain the five model training runs that we will compare.
# MAGIC
# MAGIC The experiment path is derived from the current notebook path to avoid hardcoding the user workspace path.

# COMMAND ----------

experiment_path = notebook_path.rsplit("/", 1)[0] + "/MLflow_Experiment"

print("MLflow Experiment Path:")
print(experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 61: Create the MLflow Experiment
# MAGIC
# MAGIC We create a dedicated MLflow Experiment for the Calories Burnt Prediction project.
# MAGIC
# MAGIC All model training runs performed in this phase will be associated with this experiment.

# COMMAND ----------

mlflow.set_experiment(experiment_path)

print("MLflow experiment configured successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 62: Verify the MLflow Experiment
# MAGIC
# MAGIC We verify that MLflow is configured to use the experiment created for this project.

# COMMAND ----------

print("Experiment:", experiment_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 63: Track the Linear Regression Experiment
# MAGIC
# MAGIC We start an MLflow run for the Linear Regression model.
# MAGIC
# MAGIC An MLflow run represents one execution of a machine learning experiment.
# MAGIC
# MAGIC During this run, we will record:
# MAGIC - Model parameters
# MAGIC - Evaluation metrics
# MAGIC - The trained model

# COMMAND ----------

with mlflow.start_run(run_name="Linear_Regression"):

    # Create the model
    linear_model = LinearRegression()

    # Create the complete pipeline
    linear_pipeline = Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", linear_model)
        ]
    )

    # Train the model
    linear_pipeline.fit(X_train, y_train)

    # Generate predictions
    linear_predictions = linear_pipeline.predict(X_test)

    # Calculate metrics
    linear_mae = mean_absolute_error(y_test, linear_predictions)
    linear_rmse = np.sqrt(
        mean_squared_error(y_test, linear_predictions)
    )
    linear_r2 = r2_score(y_test, linear_predictions)

    # Log metrics
    mlflow.log_metric("MAE", linear_mae)
    mlflow.log_metric("RMSE", linear_rmse)
    mlflow.log_metric("R2", linear_r2)

    # Log the model
    mlflow.sklearn.log_model(
        linear_pipeline,
        name="linear_regression_model"
    )

    print("Linear Regression")
    print(f"MAE  : {linear_mae:.4f}")
    print(f"RMSE : {linear_rmse:.4f}")
    print(f"R2   : {linear_r2:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 64: Define Models and Their Hyperparameters
# MAGIC
# MAGIC We define the five candidate regression models along with their relevant hyperparameters.
# MAGIC
# MAGIC These configurations will be used to train and track each model as a separate MLflow run.
# MAGIC
# MAGIC The parameters will be logged to MLflow so that we can later understand exactly how each model was trained.

# COMMAND ----------

mlflow_models = {
    "Linear Regression": {
        "model": LinearRegression(),
        "params": {}
    },

    "Decision Tree": {
        "model": DecisionTreeRegressor(
            max_depth=10,
            random_state=42
        ),
        "params": {
            "max_depth": 10,
            "random_state": 42
        }
    },

    "Random Forest": {
        "model": RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        ),
        "params": {
            "n_estimators": 200,
            "max_depth": 10,
            "random_state": 42
        }
    },

    "Gradient Boosting": {
        "model": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        ),
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 5,
            "random_state": 42
        }
    },

    "XGBoost": {
        "model": XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        ),
        "params": {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "random_state": 42
        }
    }
}

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 65: Track All Five Models with MLflow
# MAGIC
# MAGIC Each model is trained inside its own MLflow run.
# MAGIC
# MAGIC For every run, we:
# MAGIC
# MAGIC 1. Create the preprocessing and model pipeline.
# MAGIC 2. Log the model parameters.
# MAGIC 3. Train the model.
# MAGIC 4. Generate predictions.
# MAGIC 5. Calculate MAE, RMSE, and R².
# MAGIC 6. Log the evaluation metrics.
# MAGIC 7. Log the trained model.
# MAGIC
# MAGIC This creates a reproducible experiment history containing one run for each candidate model.

# COMMAND ----------

mlflow_results = []

for model_name, model_info in mlflow_models.items():

    with mlflow.start_run(run_name=model_name.replace(" ", "_")):

        model = model_info["model"]
        params = model_info["params"]

        # Create pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model)
            ]
        )

        # Log parameters
        if params:
            mlflow.log_params(params)

        mlflow.log_param("algorithm", model_name)

        # Train
        pipeline.fit(X_train, y_train)

        # Predict
        predictions = pipeline.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Log model
        mlflow.sklearn.log_model(
            pipeline,
            name="model"
        )

        # Save result locally for comparison
        mlflow_results.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"{model_name} completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 66: Create a Model Input Example
# MAGIC
# MAGIC A model input example provides MLflow with a representative example of the data expected by the trained model.
# MAGIC
# MAGIC MLflow can use this example to infer and store the model's input and output signature.
# MAGIC
# MAGIC The input example will use a single row from the original feature dataset.

# COMMAND ----------

input_example = X_train.iloc[[0]]

input_example

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 67: Generate the MLflow Model Signature
# MAGIC
# MAGIC The model signature describes the expected inputs and outputs of the machine learning model.
# MAGIC
# MAGIC This provides a clear contract between the deployed model and applications that will consume it.

# COMMAND ----------

from mlflow.models import infer_signature

signature = infer_signature(
    X_train,
    linear_pipeline.predict(X_train)
)

signature

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 69: Track All Five Models with Signatures
# MAGIC
# MAGIC We retrain the five candidate models and log each experiment to MLflow.
# MAGIC
# MAGIC For every run, we record:
# MAGIC
# MAGIC - Model algorithm
# MAGIC - Hyperparameters
# MAGIC - MAE
# MAGIC - RMSE
# MAGIC - R²
# MAGIC - Input example
# MAGIC - Model signature
# MAGIC - Trained model
# MAGIC
# MAGIC The model signature and input example allow MLflow and downstream serving systems to understand the expected input and output structure of the model.

# COMMAND ----------

mlflow_results_v2 = []

for model_name, model_info in mlflow_models.items():

    with mlflow.start_run(
        run_name=f"{model_name.replace(' ', '_')}_v2"
    ):

        model = model_info["model"]
        params = model_info["params"]

        # Create preprocessing + model pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", clone(preprocessor)),
                ("model", model)
            ]
        )

        # Log algorithm
        mlflow.log_param("algorithm", model_name)

        # Log model hyperparameters
        if params:
            mlflow.log_params(params)

        # Train
        pipeline.fit(X_train, y_train)

        # Predictions
        predictions = pipeline.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        r2 = r2_score(y_test, predictions)

        # Log metrics
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Create model signature
        signature = infer_signature(
            X_train,
            pipeline.predict(X_train)
        )

        # Log model with signature and input example
        mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=signature,
            input_example=input_example
        )

        # Store results for comparison
        mlflow_results_v2.append({
            "Model": model_name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })

        print(f"{model_name} completed")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 70: Compare the MLflow Experiment Results
# MAGIC
# MAGIC The metrics from the five MLflow runs are collected into a DataFrame and sorted by MAE.
# MAGIC
# MAGIC This allows us to verify that the MLflow-tracked experiments produce the same model ranking as our original model comparison.

# COMMAND ----------

mlflow_results_v2_df = pd.DataFrame(mlflow_results_v2)

mlflow_results_v2_df = mlflow_results_v2_df.sort_values(
    by="MAE",
    ascending=True
).reset_index(drop=True)

mlflow_results_v2_df

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 71: Select the Best MLflow Model
# MAGIC
# MAGIC The model with the lowest MAE is selected as the candidate model for the Model Registry.
# MAGIC
# MAGIC RMSE and R² are also considered to confirm that the selected model performs well across multiple evaluation metrics.

# COMMAND ----------

best_mlflow_model = mlflow_results_v2_df.iloc[0]

print("Selected Model:", best_mlflow_model["Model"])
print(f"MAE : {best_mlflow_model['MAE']:.4f}")
print(f"RMSE: {best_mlflow_model['RMSE']:.4f}")
print(f"R2  : {best_mlflow_model['R2']:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 72: Verify the MLflow Experiment
# MAGIC
# MAGIC The MLflow Experiment should now contain the model runs created during experimentation.
# MAGIC
# MAGIC The improved runs contain:
# MAGIC
# MAGIC - Parameters
# MAGIC - Metrics
# MAGIC - Model
# MAGIC - Input example
# MAGIC - Model signature
# MAGIC
# MAGIC These tracked experiments will provide the evidence used to select a model for registration.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 73: Configure Unity Catalog Model Registry
# MAGIC
# MAGIC MLflow 3 uses Unity Catalog as the model registry.
# MAGIC
# MAGIC We explicitly configure the registry URI so that our model will be registered in Unity Catalog.
# MAGIC

# COMMAND ----------

import mlflow

mlflow.set_registry_uri("databricks-uc")

print("Registry URI:", mlflow.get_registry_uri())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 74: Define the Registered Model Name
# MAGIC
# MAGIC We define the name that will be used for our production model in Unity Catalog.
# MAGIC
# MAGIC The model name follows the three-level Unity Catalog namespace:
# MAGIC
# MAGIC catalog.schema.model

# COMMAND ----------

CATALOG_NAME = "databricks00"
SCHEMA_NAME = "default"
MODEL_NAME = "calories_burnt_predictor"

registered_model_name = (
    f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
)

print("Registered model name:")
print(registered_model_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 75: Verify Unity Catalog Access
# MAGIC
# MAGIC We verify that the selected catalog and schema are accessible before attempting to register the model.

# COMMAND ----------

spark.sql(f"USE CATALOG {CATALOG_NAME}")
spark.sql(f"USE SCHEMA {SCHEMA_NAME}")

print("Catalog:", CATALOG_NAME)
print("Schema :", SCHEMA_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 76: Retrieve the XGBoost_v2 MLflow Run
# MAGIC
# MAGIC XGBoost_v2 was selected as our best-performing model.
# MAGIC
# MAGIC We retrieve its MLflow run so that we can locate the exact logged model that we want to register.

# COMMAND ----------

experiment = mlflow.get_experiment_by_name(experiment_path)

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="tags.mlflow.runName = 'XGBoost_v2'"
)

xgb_run_id = runs.iloc[0]["run_id"]

print("XGBoost_v2 Run ID:")
print(xgb_run_id)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 77: Retrieve the XGBoost Logged Model
# MAGIC
# MAGIC We retrieve the MLflow LoggedModel associated with the XGBoost_v2 run.
# MAGIC
# MAGIC The LoggedModel represents the packaged model that was created during our MLflow experiment.

# COMMAND ----------

logged_models = mlflow.search_logged_models(
    experiment_ids=[experiment.experiment_id]
)

print("Available columns:")
print(logged_models.columns.tolist())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 78: Identify the XGBoost_v2 Logged Model
# MAGIC
# MAGIC We filter the LoggedModels returned by MLflow using the `source_run_id` of our XGBoost_v2 experiment run.
# MAGIC
# MAGIC This identifies the exact LoggedModel produced by the XGBoost_v2 run.

# COMMAND ----------

xgb_logged_model = logged_models[
    logged_models["source_run_id"] == xgb_run_id
]

xgb_logged_model[
    ["model_id", "name", "source_run_id", "status"]
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 79: Extract the Logged Model ID
# MAGIC
# MAGIC We extract the unique LoggedModel ID of XGBoost_v2.
# MAGIC
# MAGIC This ID will be used as the source when registering the model in Unity Catalog.

# COMMAND ----------

xgb_model_id = xgb_logged_model.iloc[0]["model_id"]

print("XGBoost Logged Model ID:")
print(xgb_model_id)

# COMMAND ----------

xgb_logged_model[
    ["model_id", "name", "source_run_id", "status"]
]

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 80: Register the XGBoost Model in Unity Catalog
# MAGIC
# MAGIC The XGBoost_v2 model was selected as the best-performing model during our MLflow experiment.
# MAGIC
# MAGIC The LoggedModel is in the `READY` state, so we can now register it in Unity Catalog.
# MAGIC
# MAGIC Registering the model gives it a persistent name and version that can be used throughout the remaining MLOps lifecycle.

# COMMAND ----------

model_uri = f"models:/{xgb_model_id}"

registered_model = mlflow.register_model(
    model_uri,
    registered_model_name
)

print("Registered Model:")
print(registered_model)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 81: Verify the Registered Model
# MAGIC
# MAGIC The model has been registered in Unity Catalog.
# MAGIC
# MAGIC We now retrieve the available versions of the registered model to verify that Version 1 was created successfully.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

model_versions = client.search_model_versions(
    filter_string=f"name='{registered_model_name}'"
)

for version in model_versions:
    print("Model Name :", version.name)
    print("Version    :", version.version)
    print("Status     :", version.status)
    print("Source     :", version.source)
    print("Run ID     :", version.run_id)
    print("-" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 82: Retrieve Model Version 1
# MAGIC
# MAGIC We retrieve Version 1 of the registered model.
# MAGIC
# MAGIC This version represents the XGBoost model that we selected based on our MLflow experiment results.

# COMMAND ----------

model_version = client.get_model_version(
    name=registered_model_name,
    version="1"
)

print("Model Name :", model_version.name)
print("Version    :", model_version.version)
print("Status     :", model_version.status)
print("Source     :", model_version.source)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 83: Document the Registered Model
# MAGIC
# MAGIC We add a description to the registered model so that anyone viewing the model in Unity Catalog can understand its purpose and origin.

# COMMAND ----------

client.update_registered_model(
    name=registered_model_name,
    description=(
        "XGBoost regression model for predicting calories burnt "
        "during exercise. Selected from five candidate regression "
        "models using MAE, RMSE, and R2 evaluation metrics. "
        "Tracked using MLflow and registered in Unity Catalog."
    )
)

print("Model description updated successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 84: Add Metadata to Model Version 1
# MAGIC
# MAGIC We add metadata to Version 1 to document why this particular model version was selected.

# COMMAND ----------

client.set_model_version_tag(
    name=registered_model_name,
    version="1",
    key="model_type",
    value="XGBoost"
)

client.set_model_version_tag(
    name=registered_model_name,
    version="1",
    key="selection_reason",
    value="Best performance among five evaluated regression models"
)

client.set_model_version_tag(
    name=registered_model_name,
    version="1",
    key="experiment",
    value="Calories_Burnt_MLOps"
)

print("Model version metadata added successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 85: Define the Production Model URI
# MAGIC
# MAGIC The model has been registered in Unity Catalog as `databricks00.default.calories_burnt_predictor`.
# MAGIC
# MAGIC Version 1 contains the XGBoost model selected from our MLflow experiment.
# MAGIC
# MAGIC We define the model URI that will be used to load this registered version.

# COMMAND ----------

production_model_uri = (
    "models:/databricks00.default.calories_burnt_predictor/1"
)

print("Production Model URI:")
print(production_model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 86: Load the Registered Model
# MAGIC
# MAGIC We load Version 1 of the registered model from Unity Catalog using MLflow.
# MAGIC
# MAGIC This verifies that the model stored in the registry can be loaded independently of the original training code.

# COMMAND ----------

loaded_model = mlflow.pyfunc.load_model(
    production_model_uri
)

print("Registered model loaded successfully.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 87: Create a Sample Prediction Request
# MAGIC
# MAGIC We create a sample input containing the features required by the registered model.
# MAGIC
# MAGIC This simulates a new exercise record arriving after the model has been deployed.

# COMMAND ----------

sample_input = pd.DataFrame({
    "Gender": ["male"],
    "Age": [25],
    "Height": [175],
    "Weight": [70],
    "Duration": [30],
    "Heart_Rate": [120],
    "Body_Temp": [39.0]
})

sample_input

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 88: Create a Schema-Compatible Prediction Request
# MAGIC
# MAGIC The registered model contains an MLflow model signature that expects several numerical features as floating-point values.
# MAGIC
# MAGIC We therefore construct the prediction request using the data types defined by the model signature.

# COMMAND ----------

sample_input = pd.DataFrame({
    "Gender": ["male"],
    "Age": [25],
    "Height": [175.0],
    "Weight": [70.0],
    "Duration": [30.0],
    "Heart_Rate": [120.0],
    "Body_Temp": [39.0]
})

print(sample_input)
print("\nData types:")
print(sample_input.dtypes)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 89: Generate a Prediction Using the Registered Model
# MAGIC
# MAGIC We pass the schema-compatible input to Version 1 of the registered model.
# MAGIC
# MAGIC The model will automatically perform the preprocessing and generate the predicted calories burnt.

# COMMAND ----------

prediction = loaded_model.predict(sample_input)

print("Predicted Calories Burnt:", prediction[0])

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 90: Understand the Model Serving Endpoint
# MAGIC
# MAGIC A Model Serving endpoint provides a real-time interface for our registered machine learning model.
# MAGIC
# MAGIC Instead of loading the model manually using Python, an application can send a prediction request to the endpoint.
# MAGIC
# MAGIC The endpoint will:
# MAGIC
# MAGIC 1. Receive the input data.
# MAGIC 2. Validate the input against the model signature.
# MAGIC 3. Execute the registered model.
# MAGIC 4. Return the predicted calories burnt.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 91: Define the Model Serving Endpoint
# MAGIC
# MAGIC We define a unique name for the serving endpoint.
# MAGIC
# MAGIC The endpoint will serve Version 1 of our registered `calories_burnt_predictor` model.

# COMMAND ----------

endpoint_name = "calories-burnt-predictor"

print("Serving endpoint:", endpoint_name)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 92: Create the Model Serving Endpoint
# MAGIC
# MAGIC We create a real-time Model Serving endpoint for the registered model.
# MAGIC
# MAGIC The endpoint will use Version 1 of `databricks00.default.calories_burnt_predictor`.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 93: Understand the Model Serving Endpoint
# MAGIC
# MAGIC The registered XGBoost model is now deployed as a real-time serving endpoint.
# MAGIC
# MAGIC The endpoint accepts prediction requests and returns predictions from Version 1 of the registered model.
# MAGIC
# MAGIC This separates model inference from the training notebook.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 94: Identify the Model Serving Endpoint URL
# MAGIC
# MAGIC The serving endpoint is now ready.
# MAGIC
# MAGIC The endpoint exposes an HTTP API that applications can use to send prediction requests to our deployed model.
# MAGIC
# MAGIC We will use this URL to communicate with the deployed XGBoost model.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 95: Send a Prediction Request to the Serving Endpoint
# MAGIC
# MAGIC We send a new exercise record to the deployed model through its REST API.
# MAGIC
# MAGIC The request contains the features expected by the model signature.
# MAGIC
# MAGIC This tests whether our deployed model can perform real-time inference.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 96: Verify the Prediction Response
# MAGIC
# MAGIC The serving endpoint should return the prediction generated by Version 1 of the registered XGBoost model.
# MAGIC
# MAGIC The returned value represents the predicted number of calories burnt for the supplied exercise record.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 97: Compare Endpoint and Local Model Predictions
# MAGIC
# MAGIC We compare the prediction returned by the deployed serving endpoint with the prediction previously generated by loading the registered model directly with MLflow.
# MAGIC
# MAGIC This verifies that the deployed endpoint is serving the same model version that we tested locally.

# COMMAND ----------

local_prediction = loaded_model.predict(sample_input)[0]

endpoint_prediction = 233.57042646484375

print("Local Model Prediction    :", local_prediction)
print("Serving Endpoint Prediction:", endpoint_prediction)
print("Difference                :", abs(
    local_prediction - endpoint_prediction
))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Step 98: Test Multiple Prediction Inputs
# MAGIC
# MAGIC We test the deployed model with multiple exercise records rather than relying on a single prediction.
# MAGIC
# MAGIC This helps verify that the serving endpoint can handle different input values and consistently return predictions.
# MAGIC
# MAGIC test1
# MAGIC
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {
# MAGIC       "Gender": "male",
# MAGIC       "Age": 25,
# MAGIC       "Height": 175.0,
# MAGIC       "Weight": 70.0,
# MAGIC       "Duration": 30.0,
# MAGIC       "Heart_Rate": 120.0,
# MAGIC       "Body_Temp": 39.0
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC
# MAGIC test 2
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {
# MAGIC       "Gender": "female",
# MAGIC       "Age": 30,
# MAGIC       "Height": 165.0,
# MAGIC       "Weight": 60.0,
# MAGIC       "Duration": 45.0,
# MAGIC       "Heart_Rate": 135.0,
# MAGIC       "Body_Temp": 38.5
# MAGIC     }
# MAGIC   ]
# MAGIC }
# MAGIC
# MAGIC test 3
# MAGIC {
# MAGIC   "dataframe_records": [
# MAGIC     {
# MAGIC       "Gender": "male",
# MAGIC       "Age": 40,
# MAGIC       "Height": 180.0,
# MAGIC       "Weight": 85.0,
# MAGIC       "Duration": 60.0,
# MAGIC       "Heart_Rate": 145.0,
# MAGIC       "Body_Temp": 39.2
# MAGIC     }
# MAGIC   ]
# MAGIC }