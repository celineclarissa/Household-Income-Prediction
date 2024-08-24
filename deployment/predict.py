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