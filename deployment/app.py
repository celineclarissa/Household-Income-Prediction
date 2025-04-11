'''
==========================================================================================================================================

Household Income Prediction

Name: Celine Clarissa

Original Dataset: https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis/data

Deployment: https://huggingface.co/spaces/celineclarissa/Milestone2_Household_Income_Prediction

GitHub: https://github.com/celineclarissa/Household-Income-Prediction


Background

As a data scientist at a governmental organization focused on regional economic research, my role involves analyzing household income
as a key indicator of economic conditions. The organization aims not only to study historical data but also to develop predictive models
to estimate household income from new, unseen data. This requires a deep understanding of the factors that influence income in order to
build reliable and informative models to support effective planning.


Problem Statement and Objectives

As a data scientist, it is essential to be able to train, test, fine-tune, and evaluate machine learning models—especially when those
models are used to support real-world decision-making. In this case, predicting household income allows the organization to better
understand the region’s economic condition and develop strategies for urban development and resource planning.

The process begins with exploratory data analysis (EDA) to uncover key patterns and relationships, followed by feature engineering to
enhance model performance. Multiple algorithms—including Logistic Regression, K-Nearest Neighbors, SVR, Decision Trees, Random Forest,
AdaBoost, and Gradient Boosting—are evaluated using cross-validation and MAE as the primary metric. The most promising model is then
fine-tuned using GridSearch to improve accuracy, with a target MAE of under 5,000. The final model is deployed on Hugging Face, along
with a web app that includes both prediction capabilities and interactive EDA visualizations.

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