# Household Income Prediction
This repository showcases how I obtained insights based on analysis on factors correlating to household income for a governmental organization focusing on economic conditions and developed a model that can predict household income using the SVR model algorithm. Achieved MAE score of 4,448 and evaluated the model’s strengths and weaknesses. The analysis and model is deployed on Hugging Face for effective use.

## Background
I am a data scientist at a governmental organization that is currently researching about the economic condition of a region. The organization decided to do research about economic condition based on household income. Not only does the organization want to research using previous records, the organization also wants to predict household income from new, unseen data. From this, being a data scientist at this organization, it is crucial to understand what factors affect household income, and then create a model that can make predictions. 

## Problem Statement
As a data scientist, it is crucial to have skills of training, testing, tuning, and evaluating a model because the organization can use model to predict household income. Then, the household income predictions can be used to predict economic condition in the region. The organization can later plan strategies regarding economic development in the city.

This can be done by using data. After analyzing the  in EDA process, data scientist will then do feature engineering towards data. Then, data scientist will do cross validation with Logistic Regression, KNN, SVR, Decision Tree, Random Forest, AdaBoost, and GradientBoost model algorithms using MAE score. The best model set with default parameters will then be tuned with GridSearch so that the best model can be attained. The model is aimed to have an MAE score of less than 5,000 and then deployed on Hugging Face for effective use after 7 working days. Webapp where model is deployed will also feature a page for EDA. 

## Methods Used
* Exploratory Data Analysis
* Data Visualization
* Machine Learning (Regression)

## Technologies
* Python
* Pandas
* Matplotlib
* Scikit-Learn
* Phi-K
* Streamlit
* Hugging Face.

## Featured Links
Original Dataset: [Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis/data)

Deployment: [Hugging Face](https://huggingface.co/spaces/celineclarissa/Milestone2_Household_Income_Prediction)
