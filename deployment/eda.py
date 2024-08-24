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
import matplotlib.pyplot as plt
import seaborn as sns

# set page title
st.set_page_config(
    page_title = 'Milestone 2'
)

# make function run()
def run():
    # make title
    st.title('Household Income Data EDA')

    # make description
    st.write('This page was made to predict household income.')

    # insert image
    st.image('https://ebizz.co.uk/wp-content/uploads/2024/05/38k-After-Tax.jpg', caption='Annual Income (https://ebizz.co.uk/wp-content/uploads/2024/05/38k-After-Tax.jpg)')

    # Membuat garis lurus
    st.markdown('---')

    # Show dataframe
    st.write('### Dataset')
    df = pd.read_csv('data.csv')

    # show dataset
    st.dataframe(df)

    # make border
    st.write('')
    st.markdown('---')
    st.write('')

    # EDA 1
    # make title
    st.write('### Income Distribution')

    # make visualization
    fig1 = plt.figure(figsize=[15, 5])
    plt.subplot(1,2,1)
    sns.histplot(df['Income'], kde=True, bins = 30)
    plt.title('Income Histogram')
    # show visualization
    st.pyplot(fig1)

    # show insight for EDA 1
    st.write('From the histogram, it can be seen that the data is very skewed. More than half the data have income of less than USD100,000 when maximum value is near USD1,000,000.')



    # make border
    st.markdown('---')



    # EDA 2
    # make title
    st.write('### Age: Distribution, Grouping, Income for Each Age Group')

    # create histogram
    fig2_1 = plt.figure(figsize=[15, 5])
    sns.histplot(df['Age'], kde=True, bins = 30)
    plt.title('Age Histogram')
    # show visualization
    st.pyplot(fig2_1)

    # show insight based on histogram
    st.write("From the histogram, it can be understood that the data is distributed equally, and it has a pattern that goes on continuously throughout data.")

    # make looping to group data based on age
    age_group = []
    for i in df['Age']:
        if i in range (0, 14):
            age_group.append('Children')
        elif i in range(15, 24):
            age_group.append('Youth')
        elif i in range(25, 64):
            age_group.append('Adult')
        else:
            age_group.append('Senior')
    df['age_group'] = age_group

    # show age group distribution
    fig2_2 = plt.figure(figsize=[15, 5])
    agegroup_counts = df['age_group'].value_counts(dropna=False)
    agegroup_counts.plot.pie(autopct='%1.1f%%', shadow=True)
    plt.title('Age Distribution Pie Chart')
    plt.axis('equal')
    st.pyplot(fig2_2)

    # show insight based on pie chart
    st.write("After grouping data based on age group with [reference](https://www.statcan.gc.ca/en/concepts/definitions/age2), it was found that the majority of household have primary member who are adults. Those who are seniors and in their youths combined make up around a quarter of data.")

    # show income for each age group
    # create new table 
    table_eda2 = df.loc[df.age_group != '0', ['age_group', 'Income']].groupby('age_group').mean().sort_values('Income').reset_index()
    # create bar chart
    fig2_3 = plt.figure(figsize=[15, 5])

    table_eda2.plot(kind='bar', x='age_group', y='Income', xlabel='Age Group')
    plt.title('Income for Each Age Group')
    # show visualization
    st.pyplot(fig2_3)

    # show insight for EDA 2
    st.write("***From the bar chart, it can be understood that there is significant difference in average annual household income based on primary member's age group.*** Those in age group 'senior' have highest average annual income, followed by 'adult', and 'youth', although the difference is not very high. All age groups have average income in range USD700,000 to USD800,000.")



    # make border
    st.markdown('---')



    # EDA 3
    # make title
    st.write('### Work Experience: Grouping: Income for Each Group')

    # show work experience distribution
    fig3_1 = plt.figure(figsize=[15, 5])
    sns.histplot(df['Work_Experience'], kde=True, bins = 30)
    plt.title('Work Experience Histogram')
    # show visualization
    st.pyplot(fig3_1)

    # show insight based on histogram
    st.write("From the histogram, it can be understood that the data is distributed equally and the distribution has a pattern that goes continuously throughout data.")

    # income based on grouping
    # group data based on work experience
    experience_group = []
    for i in df['Work_Experience']:
        if 0 < i <= 12: # 0 is min value in data, 12 is first quartile in data
            experience_group.append('Little experience')
        elif 12 < i <= 25: # 12 is first quartile in data, 25 is second quartile in data
            experience_group.append('Intermediate')
        elif 25 < i <= 37: # 25 is second quartile in data, 37 is thirs quartile in data
            experience_group.append('Moderately experienced')
        else:
            experience_group.append('Very experienced')
    df['experience_group'] = experience_group
    # create new table
    table_eda3 = df.loc[df.experience_group != '0', ['experience_group', 'Income']].groupby('experience_group').mean().sort_values('Income').reset_index()
    # create bar chart
    fig3_2 = plt.figure(figsize=[15, 5])
    table_eda3.plot(kind='bar', x='experience_group', y='Income', xlabel='')
    plt.title('Income for Each Age Group')
    # show visualization
    st.pyplot(fig3_2)

    # show insight for EDA 3
    st.write("***From the bar chart, it can be understood that there is a significant difference in average annual household income between different work experience groups.*** Households with primary members who have little experience the highest average income. Meanwhile, households with lowest average annual income are households with primary members who are very experienced. The groups have an order with the highest being 'very experienced' and lowest being 'little experience'. However, the order of the groups is not the same for household income context.")



    # make border
    st.markdown('---')



    # EDA 4
    # make title
    st.write('### Household Size: Distribution, Grouping, Income for Each Group')

    # show household size distribution
    fig4_1 = plt.figure(figsize=[15, 5])
    sns.histplot(df['Household_Size'], kde=True, bins = 30)
    plt.title('Household Size Histogram')
    # show visualization
    st.pyplot(fig4_1)

    # show insight based on histogram
    st.write("From the histogram, it can be understood that the 'Household_Size' are distributed quite equally, with all of them having around 1,400 data.")

    # income based on grouping
    # group data based on household size
    householdsize_group = []
    for i in df['Household_Size']:
        if 0 < i <= 2: # 0 is minimum value in data, 2 is first quartile in data
            householdsize_group.append('Small household')
        elif 2 < i <= 6: # 2 is first quartile in data, 6 is third quartile in data
            householdsize_group.append('Medium household')
        else:
            householdsize_group.append('Big household')
    df['householdsize_group'] = householdsize_group
    # create new table
    table_eda4 = df.loc[df.householdsize_group != '0', ['householdsize_group', 'Income']].groupby('householdsize_group').mean().sort_values('Income').reset_index()
    # create bar chart
    fig4_2 = plt.figure(figsize=[15, 5])
    table_eda4.plot(kind='bar', x='householdsize_group', y='Income', xlabel='Household Size')
    plt.title('Income for Each Group')
    # show visualization
    st.pyplot(fig4_2)

    # show insight based on grouping
    st.write("***From the bar chart, it can be understood that there is no significant difference in average annual household income amongst groups made based on household size.*** All of them have average annual income of around USD800,000.")



    # make border
    st.markdown('---')



    # EDA 5
    # make title
    st.write('### Number of Dependents: Distribution, Grouping, Income for Each Group')

    # show number of dependents distribution
    fig5_1 = plt.figure(figsize=[15, 5])
    sns.histplot(df['Number_of_Dependents'], kde=True, bins = 30)
    plt.title('Number of Dependents Histogram')
    # show visualization
    st.pyplot(fig5_1)

    # show insight based on histogram
    st.write("From the histogram, it can be understood that the 'Number_of_Dependents' are distributed quite equally, with all of them having more than 1,500 data.")

    # income based on grouping
    # group data based on number of dependents
    ndependents_group = []
    for i in df['Number_of_Dependents']:
        if 0 < i <= 3: # 0 is minimum value in data, 3 is second quartile in data
            ndependents_group.append('Little')
        elif 3 < i <= 4: # 3 is second quartile in data, 4 is thirs quartile in data
            ndependents_group.append('Average')
        else:
            ndependents_group.append('Many')
    df['ndependents_group'] = ndependents_group
    # create new table
    table_eda5 = df.loc[df.ndependents_group != '0', ['ndependents_group', 'Income']].groupby('ndependents_group').mean().sort_values('Income').reset_index()
    # create bar chart
    fig5_2 = plt.figure(figsize=[15, 5])
    table_eda5.plot(kind='bar', x='ndependents_group', y='Income', xlabel='Dependents in Household')
    plt.title('Income for Each Group')
    # show visualization
    st.pyplot(fig5_2)

    # show insight based on grouping
    st.write("***From the bar chart, it can be understood that there is no significant difference in average annual household income amongst groups made based on number of dependents in household.*** All of them have average annual income of around USD800,000.")



    # make border
    st.markdown('---')



    # EDA 6
    # make title
    st.write('### Distribution of Categorical Columns in Data')
    # define choice_eda4 based on user input
    choice_eda6 = st.selectbox('Choose feature:', ['Education_Level', 'Occupation', 'Location', 'Marital_Status', 'Employment_Status', 'Homeownership_Status', 'Type_of_Housing', 'Gender', 'Primary_Mode_of_Transportation'])
    # create dictionary for visualization purposes
    dict_eda6 = {'Education_Level': ["Master's", 'High School', "Bachelor's", 'Doctorate'],
             'Occupation': ['Technology', 'Finance', 'Others', 'Education', 'Healthcare'],
             'Location': ['Urban', 'Rural', 'Suburban'],
             'Marital_Status': ['Married', 'Single', 'Divorced'],
             'Employment_Status': ['Full-time', 'Self-employed', 'Part-time'],
             'Homeownership_Status': ['Own', 'Rent'],
             'Type_of_Housing': ['Apartment', 'Single-family home', 'Townhouse'],
             'Gender': ['Male', 'Female'],
             'Primary_Mode_of_Transportation': ['Public transit', 'Biking', 'Car', 'Walking']}
    # create new table based on user input
    table_eda6 = df[choice_eda6].value_counts(dropna=False)
    # create pie chart
    fig6 = plt.figure(figsize=[15, 5])
    plt.pie(table_eda6, autopct='%1.1f%%', labels=dict_eda6[choice_eda6], shadow=True)
    plt.title(f'{choice_eda6} Distribution Pie Chart')
    plt.axis('equal')
    # show visualization
    st.pyplot(fig6)

    # show insight for EDA 6
    if choice_eda6 == 'Education_Level':
        st.write("***The distribution in household primary member's education level is not equal.*** Many of them developed achieved highest education level of Master's degree, having around 40%, followed by High School diploma and Bachelor's degree with similar percentages. Lastly, only 5% of household's primary member have attained a Doctorate's degree. ")
    elif choice_eda6 == 'Occupation':
        st.write("***The distribution in household primary member's occupation field is not equal.*** Many of them developed a career in finance and technology with similar percentages. The three other fields are distributed quite equally with each having around 15%, half of those in finance and technology.")
    elif choice_eda6 == 'Location':
        st.write("***From the pie chart, it can be understood that the distribution in household location is not equal.*** There are far more household located in urban areas than those in rural and suburban areas, having percentage of around 70%. ")
    elif choice_eda6 == 'Marital_Status':
        st.write("***From the pie chart, it can be understood that the distribution in marital status of household's primary member is not equal.*** Half of the data have primary household members who are married. Meanwhile, only nearly a tenth of data have primary household members are divorced.")
    elif choice_eda6 == 'Employment_Status':
        st.write("***From the pie chart, it can be understood that the distribution in employment status of household's primary member is not equal.*** Half of the data have primary household members who are employed full-time. Meanwhile, only nearly a fifth of data have primary household members are employed part-time.")
    elif choice_eda6 == 'Homeownership_Status':
        st.write("***From the pie chart, it can be understood that the distribution in homeownership status is not equal.*** Between owning and renting a place, more households do the first choice with ratio of around 60:40.")
    elif choice_eda6 == 'Type_of_Housing':
        st.write("***From the pie chart, it can be understood that the distribution in type of housing is not equal.*** Households in apartments and single-family homes almost have the same amount. But, those in townhouses are only half the amount of household in apartments.")
    elif choice_eda6 == 'Gender':
        st.write("***From the pie chart, it can be understood that the distribution between household's primary member's gender in data is almost equal.*** The ratio is nearly 50:50, but there are a little more primary household members who are males with diffreence of only 2.4%.")
    else:
        st.write("***From the pie chart, it can be understood that the primary mode of transportation is not distributed evenly in  data.*** More primary members of household prefer to commute with public transit, followed by biking, car, and walking. The difference is quite significant where the percentage of primary household members who walks are almost a quarter of those who use public transit.")



    # make border
    st.markdown('---')



    # EDA 7
    # make title
    st.write('### Income Based on Location, Employment Status, and Homeownership Status')
    # define choice_eda7 based on user input
    choice_eda7 = st.selectbox('Choose feature: ', ['Education_Level', 'Location', 'Marital_Status', 'Homeownership_Status'])
    # create new table base don user input
    table_eda7 = df.loc[df.Employment_Status != '0', [choice_eda7, 'Income']].groupby(choice_eda7).mean().sort_values('Income').reset_index()
    # create bar chart
    fig7 = plt.figure(figsize=[15, 5])
    table_eda7.plot(kind='bar', x=choice_eda7, y='Income', ylabel='Income', legend=False)
    plt.title(f'Income Based on {choice_eda7}')
    # show visualization
    st.pyplot(fig7)

    # show insight for EDA 7
    if choice_eda7 == 'Education_Level':
        st.write("***From the bar graph, it can be understood that there is significant difference in average annual household income based on primary household member's education level.*** Households with primary members who attained highest education level of high school degree have the highest average annual household income. Meanwhile, households with lowest average annual household income are households with primary members who attained highest education level of Doctorate's degree. Education level have an order with the highest being Doctorate and lowest being High School. However, the order of education level is not the same for household income context.")
    elif choice_eda7 == 'Location':
        st.write("***From the bar graph, it can be understood that there is significant difference in average annual household income based on residential location.*** Households located in rural areas have the highest average annual income amongst other settings. Meanwhile, those in urban areas have the lowest average annual household income.")
    elif choice_eda7 == 'Marital_Status':
        st.write("***From the bar graph, it can be understood that there is no significant difference in average income based on primary household member's marital status.*** All three categories have average annual household income of around USD800,000.")
    else:
        st.write("***From the bar graph, it can be understood that there is significant difference in average income based on homeownership status.*** Households that are still on rent have higher average annual household income than those who own a home with difference of around USD200,000.")

    # make border
    st.markdown('---')

    # make title
    st.write("### Business Insight")
    # show important points
    st.write("The income of households in data aren't distributed equally. Most of the households have income of less than $100,000 when the maximum values is near $1,000,000.")
    st.write("Age affects income. Those in age group 'senior' have highest average annual income, followed by 'adult', and 'youth', although the difference is not very high. ")
    st.write("Education level, location, and homeownership status have significant effect on income.")
    # make subheader
    st.write("### Conclusion")
    # show conclusion
    st.write("***Data scientist recommends for governmental organization to focus on optimizing controllable variables that affect household income***, such as education level. The organization can allocate funds and give scholarships to students with potential so that they can pursue higher education, and therefore get higher chance to pursue a great career and income.")


# execute file
if __name__=='__main__':
    run()