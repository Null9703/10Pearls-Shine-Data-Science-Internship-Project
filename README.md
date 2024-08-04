# 10Pearls-Shine-Data-Science-Internship-Project
Data Science Project on the Telco Customer Churn Dataset.

## Technologies Used
1. Python
2. Scikit Learn
3. Pandas
4. Seaborn
5. Matplotlib
6. SQLite

## Module 1: Python
Goal: Process the data and analyse it.
### Step 1
Read data from excel file into a Pandas DataFrame.
![image](https://github.com/user-attachments/assets/6808c78d-82a0-4e44-a72a-383d6dd10e64)
### Step 2
Process data by:
1. checking for missing values.
2. checking for duplicates.
3. ensuring correct data types of DataFrame columns.
4. One Hot Encoding
### Step 3
Exploratory Data Analysis (EDA) using a variety of plots including box plots, bar plots, scatter plots, and heat maps.
![image](https://github.com/user-attachments/assets/95ae6581-c33d-47c2-b8b5-7937d7d5a086)
![image](https://github.com/user-attachments/assets/0743c154-8e4c-4ca4-af55-abff11e5f0e9)
### Step 4
Feature Engineering by generating Polynomial Features.
### Step 5
Evaluating feature importance using DecisionTreeRegressor's feature_importance_ attribute.

## Module 2: AI Algorithm
Goal: Model training and evaluation.
### Step 1
Perform a train-test split over our data set.
### Step 2
Scale our split data set.
### Step 3
Specify models to use:
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Support Vector Machine
### Step 4
Apply K-Fold cross-validation using 10 folds to get an out-of-sample score for each model.
### Step 5
Train our models on the training sets and test them on the testing sets.
### Step 6
Generate different metrics for each model including: accuracy, precision, recall, f1, and roc-auc.
### Step 7
Generate Confusion Matrix for each model.
### Step 8
Generate ROC curve for each model.
### Step 9
Generate Precision-Recall curve for each model.
### Step 10
Generate a classification report on each model.
### Step 11
Re-train each model on the entire data set.
### Step 12
Selecting the best model based on all their evaluation metrics.
Our best two models are Logistic Regression and Support Vector Machine.
### Step 13
GridSearch is performed on both models. Logistic Regression's best_estimator_ performs slightly better than SVM, thus we use the former for our predictions.
![image](https://github.com/user-attachments/assets/e747de4a-f20a-43f2-8ba5-9ae0c5eedbb0)
### Step 14
Model Interpretation using SHAP reveals that the most important variable to the model was tenure (and tenure^2) followed closely by contract_Two Year (an interaction term).
![image](https://github.com/user-attachments/assets/904ced29-6807-4568-907b-63db32082ed1)
### Step 15
Storing the best logistic regression model in a pickle file.

## Module 3: SQL
Goal: Design and implement a database for our dataset and query data to answer questions.
### Step 1
Design a database schema and implement it in a database management system (I chose SQLite).
### Step 2
Move data and predictions to the database.
### Step 3
Run simpler SQL queries to calculate descriptive statistics and more.
### Step 4
Run more advanced queries such as using Window Functions.
### Step 5
Report your findings for this module.

## Module 4: Model Deployment and API Consumption
Goal: Develop a frontend web application for your model using Flask. Furthermore, develop an API endpoint as well.
### Step 1
Load the model, its scaler, and polynomial features objects from pickle files.
### Step 2
Develop API and web application using Flask.
### Step 3
Prepare an html page to display as the frontend for your web application.
### Step 4
Testing web application frontend to make predictions.
![image](https://github.com/user-attachments/assets/324b1a2c-824c-4324-a389-09818ca02905)
### Step 5
Testing API using Postman
![image](https://github.com/user-attachments/assets/b7da4ea5-9801-4641-9c20-768c721d2bc8)
