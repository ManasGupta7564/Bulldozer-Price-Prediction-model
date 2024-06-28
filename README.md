# Bulldozer-Price-Prediction-model

# Predicting the Sale Price of Bulldozers using Machine Learning

In this notebook , we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.

## 1. Problem Defination
 > How well can we predict the future sale price of a bulldozer , given its characteristics and previous examples of how much similar bulldozers have been sold for ? 
## 2. Data
The data is download from the kaggle notebook for Bulldozers competition:https://www.kaggle.com/c/bluebook-for-bulldozers/data

There are 3 main datasets: 

* Train.csv is the training set, which contains data through the end of 2011.
* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
* Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.

## 3. Evaluation

The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.

For more on the evaluation of this project check : 
https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation

**Note :** The goal for most regression evalution metrics is to minimize the error . For example , our goal for this project will be to build a mlachine learning model whic minimises RMSLE.


## 4. Features

Kaggle provides a data dictionary detailing all of the featurews of the dataset. You can view this data dictionary on google sheets: 

# Key Steps

## Importing Libraries
Essential libraries like pandas, numpy, matplotlib, and scikit-learn are imported.

## Loading the Data
The dataset is loaded from a CSV file using `pandas.read_csv()`.

## Exploratory Data Analysis (EDA)
Basic EDA is performed to understand the dataset, including inspecting the first few rows and checking for missing values.

## Data Cleaning
Handling missing data by filling in missing values or dropping columns with significant amounts of missing data.

## Feature Engineering
Converting columns with object data type to categorical data types and extracting useful features from existing columns.

## Splitting the Data
The data is split into training and validation sets using `train_test_split` from scikit-learn.

## Model Selection and Training
Different machine learning models are trained, including `RandomForestRegressor`, with hyperparameter tuning and cross-validation.

## Model Evaluation
The performance of the models is evaluated using metrics such as Mean Absolute Error (MAE).

## Feature Importance
Analyzing feature importance to understand which features have the most significant impact on the model's predictions.

## Final Thoughts and Future Work
Suggestions for further model improvements and other machine learning models to try, such as CatBoost or XGBoost.

# Results
The trained models are evaluated based on their performance, and feature importance is analyzed to understand the impact of different features on the auction prices.

# Future Work
- Experiment with other machine learning models like CatBoost and XGBoost
- Further feature engineering to extract more meaningful features
- Hyperparameter tuning for better model performance

