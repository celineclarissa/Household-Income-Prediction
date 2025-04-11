# Household Income 🏠
This project investigates the relationship between various factors and household income, and applies a Support Vector Regression (SVR) model to make income predictions. Cross-validation was used to assess model stability, resulting in a Mean Absolute Error (MAE) of 4,448. The model’s performance was thoroughly evaluated to understand its strengths and limitations. It is now deployed on Hugging Face to enable easy access and practical use.

## Background ❓
As a data scientist at a governmental organization focused on regional economic research, my role involves analyzing household income as a key indicator of economic conditions. The organization aims not only to study historical data but also to develop predictive models to estimate household income from new, unseen data. This requires a deep understanding of the factors that influence income in order to build reliable and informative models to support effective planning.

## Problem Statement 🔍
As a data scientist, it’s essential to be able to train, test, fine-tune, and evaluate machine learning models—especially when those models are used to support real-world decision-making. In this case, predicting household income allows the organization to better understand the region’s economic condition and develop strategies for urban development and resource planning.

The process begins with exploratory data analysis (EDA) to uncover key patterns and relationships, followed by feature engineering to enhance model performance. Multiple algorithms—including Logistic Regression, K-Nearest Neighbors, SVR, Decision Trees, Random Forest, AdaBoost, and Gradient Boosting—are evaluated using cross-validation and MAE as the primary metric. The most promising model is then fine-tuned using GridSearch to improve accuracy, with a target MAE of under 5,000. The final model is deployed on Hugging Face, along with a web app that includes both prediction capabilities and interactive EDA visualizations.

## Methods Used 📊
* Exploratory Data Analysis
* Data Visualization
* Machine Learning (Regression)

## Technologies 👩🏻‍💻
* Python
* Pandas
* Matplotlib
* Scikit-Learn
* Phi-K
* Streamlit
* Hugging Face

## Featured Links 🔗
Original Dataset: [Kaggle](https://www.kaggle.com/datasets/stealthtechnologies/regression-dataset-for-household-income-analysis/data)

Deployment: [Hugging Face](https://huggingface.co/spaces/celineclarissa/Milestone2_Household_Income_Prediction)
