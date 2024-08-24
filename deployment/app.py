'''
==========================================================================================================================================

Household Income Prediction

Name: Celine Clarissa

Original Dataset: https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis/data

Deployment: https://huggingface.co/spaces/celineclarissa/Milestone2_Household_Income_Prediction

GitHub: https://github.com/celineclarissa/Household-Income-Prediction


Background

I am a data scientist at a governmental organization that is currently researching about the economic condition of a region. The
organization decided to do research about economic condition based on household income. Not only does the organization want to research
using previous records, the organization also wants to predict household income from new, unseen data. From this, being a data scientist
at this organization, it is crucial to understand what factors affect household income, and then create a model that can make predictions. 


Problem Statement and Objective

As a data scientist, it is crucial to have skills of training, testing, tuning, and evaluating a model because the organization can use
model to predict household income. Then, the household income predictions can be used to predict economic condition in the region. The
organization can later plan strategies regarding economic development in the city.

This can be done by using data. After analyzing the  in EDA process, data scientist will then do feature engineering towards data. Then,
data scientist will do cross validation with Logistic Regression, KNN, SVR, Decision Tree, Random Forest, AdaBoost, and GradientBoost
model algorithms using MAE score. The best model set with default parameters will then be tuned with GridSearch so that the best model
can be attained. The model is aimed to have an MAE score of less than 5,000 and then deployed on Hugging Face for effective use after 7
working days. Webapp where model is deployed will also feature a page for EDA. 

==========================================================================================================================================
'''

# import libraries
import streamlit as st
import eda
import predict

# create sidebar to navigate in between pages
navigation = st.sidebar.selectbox('Pilih halaman:', ['EDA', 'Predict'])

# make condition
if navigation == 'EDA':
    eda.run()
else:
    predict.run()