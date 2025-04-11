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
import pandas as pd
import pickle
with open('grid_best.pkl', 'rb') as file_1:
  model = pickle.load(file_1)

def run():
    # make title
    st.title('Annual Household Income Prediction')
    # insert image
    st.image('https://www.usatoday.com/gcdn/-mm-/ebd4b0b66edf6db41818323f664834cf165c4a74/c=0-15-2118-1212/local/-/media/2017/03/24/USATODAY/USATODAY/636259681350810969-GettyImages-482689547.jpg?width=1320&height=748&fit=crop&format=pjpg&auto=webp', caption='Source: Getty Images')

    # make form
    with st.form("M2_form"):

        st.write('### Insert data')

        # define each feature
        age = st.number_input('Age', min_value=18, max_value= 70, value=45)
        edu_level = st.selectbox('Education Level', ['High School', "Bachelor's", "Master's", "Doctorate"], index=2)
        occupation = st.selectbox('Occupation', ['Healthcare', 'Education', 'Technology', 'Finance', 'Others'], index=2)
        n_dependents = st.number_input('Number of Dependents', min_value=0, max_value= 5, value=2)
        loc = st.selectbox('Location', ['Urban', 'Suburban', 'Rural'], index=1)
        work_experience = st.number_input('Work Experience', min_value=0, max_value= 50, value=23)
        marital_status = st.selectbox('Marital Status', ['Single', 'Married', 'Divorced'], index=1)
        employment_status = st.selectbox('Employment Status', ['Full-time', 'Part-time', 'Self-employed'], index=0)
        household_size = st.number_input('Household Size', min_value=1, max_value= 7, value=4)
        homeownership_status = st.selectbox('Homeownership Status', ['Own', 'Rent'], index=0)
        type_of_housing = st.selectbox('Type of Housing', ['Apartment', 'Single-family home', 'Townhouse'], index=2)
        gender = st.selectbox('Gender', ['Male', 'Female'], index=1)
        mode_of_transport = st.selectbox('Mode of Transportation', ['Car', 'Public transit', 'Biking', 'Walking'], index=0)

        # make submit button
        submitted = st.form_submit_button("Submit")

    # define inference data based on inputted data
    inf_data = {
    'Age': age,
    'Education_Level': edu_level,
    'Occupation': occupation,
    'Number_of_Dependents': n_dependents,
    'Location': loc, 
    'Work_Experience': work_experience, 
    'Marital_Status': marital_status, 
    'Employment_Status': employment_status,
    'Household_Size': household_size, 
    'Homeownership_Status': homeownership_status, 
    'Type_of_Housing': type_of_housing, 
    'Gender': gender,
    'Primary_Mode_of_Transportation': mode_of_transport
}

    # make dataframe for inference data
    inf_data = pd.DataFrame([inf_data])

    # create condition
    if submitted:
        # define result using model
        result= model.predict(inf_data)
        # print result
        st.write(f'# Household Income: {round(result[0])}')
        # show balloons after submitting
        st.spinner(text='Please wait for result')
        st.balloons()

# execute file
if __name__ == '__main__':
    run()