#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:21:00 2018

@author: don
"""

# coding: utf-8

# ## Data Loading

# In[ ]:


import io
import zipfile

import pandas as pd
import pprint
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns





pp = pprint.PrettyPrinter(indent=4).pprint

data_file_path = '../data/ibm-hr-analytics-attrition-dataset.zip'
encoding = 'utf-8-sig'


data = []
with zipfile.ZipFile(data_file_path) as zfile:
    for name in zfile.namelist():
        with zfile.open(name) as readfile:
            for line in io.TextIOWrapper(readfile, encoding):
                data.append(line.replace('\n', '').split(','))

labels=['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 
       'DistanceFromHome', 'Education', 'EducationField', 'EducationField'
       "EmployeeCount","EmployeeNumber","EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement",
       "JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
       "Over18","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours",
       "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
       "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"
      ]

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
attrition_df = pd.DataFrame(data, columns=labels)
attrition_df = attrition_df.drop([0])


attrition_df


# ## Data Discovery

# In[ ]:


# attrition_df
attrition_df.head()

# len(attrition_df.columns)

not_categorical_data = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime",
    "Over18"
]

# for data in not_categorical_data:
#     print(data , ':', attrition_df[data].unique())
    
pre_categorized_data = ["Education",
"EnvironmentSatisfaction",
"JobInvolvement",
"JobSatisfaction",
"PerformanceRating",
"RelationshipSatisfaction",
"WorkLifeBalance", 
"Gender",
"JobRole",
"StockOptionLevel"
]

#finding categorized data that has no signficance, has only one data value
# for data in pre_categorized_data:
#     if ( len(attrition_df[data].unique()) <= 1):
#         print(data , ':', attrition_df[data].unique())
        
# #looking at all pre_categorized data unique values
# for data in pre_categorized_data:
#     print(data , ':', attrition_df[data].unique())
    
#finding not_categorical_data that has no signficance, has only one data value
# for data in not_categorical_data:
#     if ( len(attrition_df[data].unique()) <= 1):
#         print(data , ':', attrition_df[data].unique())


# #### Pre-Categorized Categorical Data
# 
# Education
# 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
# 
# EnvironmentSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# 
# JobInvolvement 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# 
# JobSatisfaction 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# 
# PerformanceRating 
# 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'
# 
# RelationshipSatisfaction 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
# 
# WorkLifeBalance 1 'Bad' 2 'Good' 3 'Better' 4 'Best'

# In[ ]:


# NOT CATEGORIZED CATEGORICAL DATA
post_categorical_data = [
    "Attrition",
    "BusinessTravel",
    "Department",
    "EducationField",
    "Gender",
    "JobRole",
    "MaritalStatus",
    "OverTime"
]



## CHANGING CATEGORICAL DATA TO NUMBERS
# VERSION 1
# categorical_data_values_count = {}
# for data in not_categorical_data:
#     #get counts for data
#     categorical_data_values_count[data] = attrition_df[data].value_counts()
#     #change data to number
#     attrition_df[data] = attrition_df[data].factorize()[0]
    
# attrition_df


#VERSION 2
# iris['species_num'] = iris.species.map({'Iris-setosa':0, 'Iris-versicolor':1, 'Iris-virginica':2})

attrition_df["AttritionBool"] = attrition_df.Attrition.map({"Yes": 1, "No": 0})



attrition_df["TravelRarelyBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Travel_Rarely"]
# attrition_df["Travel_Rarely"] = attrition_df.BusinessTravel.map({"Travel_Rarely": 1, "Travel_Frequently": 2, "Non-Travel": 0})



# In[ ]:


# attrition_df.columns
#


# In[ ]:


# attrition_df["TravelRarelyBool"]


# In[ ]:


# attrition_df.Travel_RarelyBool.map({'Travel_Rarely':0, np.nan:1})


# In[ ]:


attrition_df["TravelFrequentlyBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Travel_Frequently"]
attrition_df["NonTravelBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Non-Travel"]


attrition_df["SalesDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Sales"]
attrition_df["ResearchAndDevelopmentDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Research & Development"]
attrition_df["HumanResourcesDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Human Resources"]

# # attrition_df["Department"] = attrition_df.Department.map({"Sales": 0, "Research & Development": 1, "Human Resources": 2})

attrition_df["LifeScienceEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Life Sciences']
attrition_df["OtherEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Other']
attrition_df["MedicalEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Medical']
attrition_df["MarketingEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Marketing']
attrition_df["TechnicalEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Technical Degree']
attrition_df["HumanResourcesEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Human Resources']
# attrition_df["EducationField"] = attrition_df.EducationField.map({'Life Sciences': 0, 'Other': 1, 'Medical': 2, 'Marketing': 3, 'Technical Degree': 4,
# #  'Human Resources': 5})


# attrition_df["Gender"] = attrition_df.Gender.map({"Male": 0, "Female": 1})
attrition_df["Male"] = attrition_df.Gender[attrition_df.Gender == 'Male']
attrition_df["Female"] = attrition_df.Gender[attrition_df.Gender == 'Female']



attrition_df["SalesExecutiveBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Executive']
attrition_df["ResearchScientistBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Scientist']
attrition_df["LaboratoryTechnicianBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Laboratory Technician']
attrition_df["ManufacturingDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manufacturing Director']
attrition_df["HealthcareRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Healthcare Representative']
attrition_df["ManagerBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manager']
attrition_df["SalesRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Representative']
attrition_df["ResearchDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Director']
attrition_df["HumanResourcesBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Human Resources']


# # attrition_df["JobRole"] = attrition_df.JobRole.map({'Sales Executive': 0, 'Research Scientist': 1,
# #     'Laboratory Technician': 2, 'Manufacturing Director': 3, 'Healthcare Representative': 4, 'Manager': 5,
# #  'Sales Representative': 6, 'Research Director': 7, 'Human Resources': 8})



attrition_df["DivorcedBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Divorced']
attrition_df["SingleBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Single']
attrition_df["MarriedBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Married']
# # attrition_df["MaritalStatus"] = attrition_df.MaritalStatus.map({'Single': 0, 'Married': 1, 'Divorced': 2})



attrition_df["OverTime"] = attrition_df.OverTime.map({"Yes": 1, "No": 0})

# # attrition_df["Over18"] = attrition_df.Over18.map({"Y": 1, "N": 0})

# # attrition_df

# # # attrition_df.Education


# # # for cat_data in attrition_df[pre_categorized_data + post_categorical_data]:
# # # #     print(cat_data, ' : ',attrition_df[cat_data].mode)
# # #     print(cat_data)
    
# # # attrition_df[pre_categorized_data + post_categorical_data].Education.mode()
# attrition_df


# In[ ]:


# attrition_df["TravelRarelyBool"]
attrition_df["TravelRarelyBool"] = attrition_df["TravelRarelyBool"].map({"Travel_Rarely": 1, np.nan: 0})
# attrition_df["TravelRarelyBool"]

attrition_df["NonTravelBool"] = attrition_df["NonTravelBool"].map({"Non-Travel": 1, np.nan: 0})
attrition_df["TravelFrequentlyBool"] = attrition_df["TravelFrequentlyBool"].map({"Travel_Frequently": 1, np.nan: 0})

# # attrition_df["NonTravelBool"]


attrition_df["SalesDepartmentBool"] = attrition_df["SalesDepartmentBool"].map({"Sales": 1, np.nan: 0})
attrition_df["ResearchAndDevelopmentDepartmentBool"] = attrition_df["ResearchAndDevelopmentDepartmentBool"].map({"Research & Development": 1, np.nan: 0})
attrition_df["HumanResourcesDepartmentBool"] = attrition_df["HumanResourcesDepartmentBool"].map({"Human Resources": 1, np.nan: 0})


attrition_df["LifeScienceEducationBool"] = attrition_df["LifeScienceEducationBool"].map({"Life Sciences": 1, np.nan: 0})
attrition_df["OtherEducationBool"] = attrition_df["OtherEducationBool"].map({"Other": 1, np.nan: 0})
attrition_df["MedicalEducationBool"] = attrition_df["MedicalEducationBool"].map({"Medical": 1, np.nan: 0})
attrition_df["MarketingEducationBool"] = attrition_df["MarketingEducationBool"].map({"Marketing": 1, np.nan: 0})
attrition_df["TechnicalEducationBool"] = attrition_df["TechnicalEducationBool"].map({"Technical Degree": 1, np.nan: 0})
attrition_df["HumanResourcesEducationBool"] = attrition_df["HumanResourcesEducationBool"].map({"Human Resources": 1, np.nan: 0})

attrition_df["Male"] = attrition_df.Male.map({"Male": 1, np.nan: 0})
attrition_df["Female"] = attrition_df.Female.map({"Female": 1, np.nan: 0})



# attrition_df["SalesExecutiveBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Executive']
# attrition_df["ResearchScientistBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Scientist']
# attrition_df["LaboratoryTechnicianBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Laboratory Technician']
# attrition_df["ManufacturingDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manufacturing Director']
# attrition_df["HealthcareRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Healthcare Representative']
# attrition_df["ManagerBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manager']
# attrition_df["SalesRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Representative']
# attrition_df["ResearchDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Director']
# attrition_df["HumanResourcesBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Human Resources']


attrition_df["SalesExecutiveBool"] = attrition_df["SalesExecutiveBool"].map({"Sales Executive": 1, np.nan: 0})
attrition_df["ResearchScientistBool"] = attrition_df["ResearchScientistBool"].map({"Research Scientist": 1, np.nan: 0})
attrition_df["LaboratoryTechnicianBool"] = attrition_df["LaboratoryTechnicianBool"].map({"Laboratory Technician": 1, np.nan: 0})
attrition_df["ManufacturingDirectorBool"] = attrition_df["ManufacturingDirectorBool"].map({"Manufacturing Director": 1, np.nan: 0})
attrition_df["HealthcareRepresentativeBool"] = attrition_df["HealthcareRepresentativeBool"].map({"Healthcare Representative": 1, np.nan: 0})
attrition_df["ManagerBool"] = attrition_df["ManagerBool"].map({"Manager": 1, np.nan: 0})
attrition_df["SalesRepresentativeBool"] = attrition_df["SalesRepresentativeBool"].map({"Sales Representative": 1, np.nan: 0})
attrition_df["ResearchDirectorBool"] = attrition_df["ResearchDirectorBool"].map({"Research Director": 1, np.nan: 0})
attrition_df["HumanResourcesBool"] = attrition_df["HumanResourcesBool"].map({"Human Resources": 1, np.nan: 0})

# attrition_df



attrition_df["DivorcedBool"] = attrition_df["DivorcedBool"].map({"Divorced": 1, np.nan: 0})
attrition_df["SingleBool"] = attrition_df["SingleBool"].map({"Single": 1, np.nan: 0})
attrition_df["MarriedBool"] = attrition_df["MarriedBool"].map({"Married": 1, np.nan: 0})

# attrition_df.JobRole.unique()
attrition_df


# In[ ]:


#GET MODE FOR ALL CATEGORICAL DATA
attrition_df[pre_categorized_data + post_categorical_data].mode().iloc[0]


# In[ ]:


#FIND SERIES WITH NO VARIANCE/SINGLE VALUE -> THESE SERIES DO NOT HOLD PREDICTIVE VALUE AND CAN BE DROPPED
for series in attrition_df.columns:
        if (len(attrition_df[series].unique()) <= 1): print(series, ' : ', attrition_df[series].unique()) 


# In[ ]:


attrition_df = attrition_df.drop(columns=["EducationFieldEmployeeCount", "Over18", "StandardHours"])

attrition_df


# In[ ]:


#ALL NONCATEGORICAL DATA
attrition_df.head

len(list(attrition_df))

list(attrition_df)
##mean range median mode(mrm) this is to get mean, range, mode and median for all noncategorical data
series_mrm = [
    'Age',
    'DailyRate',
    'DistanceFromHome',
    'EnvironmentSatisfaction',
    'HourlyRate',
    "JobSatisfaction",
    "MonthlyIncome",
    "MonthlyRate",
    "NumCompaniesWorked",
    "PercentSalaryHike",
    "PerformanceRating",
    "RelationshipSatisfaction",
    "StockOptionLevel",
    "TotalWorkingYears",
    "TrainingTimesLastYear",
    "WorkLifeBalance",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "YearsSinceLastPromotion",
    "YearsWithCurrManager"
]



# In[ ]:


#converts all data types to floats
for series in attrition_df.columns:
    attrition_df[series] = attrition_df[series].astype(float)
    
attrition_df


# In[ ]:


#confirms that data types are no longer strings
attrition_df.applymap(type).eq(str).all()

#assert


# In[ ]:



# attrition_df[series_mrm].mean()
# attrition_df.Age.mean()

# type(attrition_df.Age.iloc[0])

#convert all data from string type to float


# attrition_df[series_mrm].mean()
# attrition_df[series_mrm].mode()
# attrition_df[series_mrm].median()
# attrition_df[series_mrm].range()

# range = attrition_df.max() - attrition_df.min()

# range = (range ^ 2) / range


# range


# ### CONFIRM NULL VALUES

# ### Change Categorical Data To Numeric

# ### Boolean Data
# 
# 'Attrition' 
# 
# Will be converted from 'yes' & 'no' to 1 & 0 respectively

# In[ ]:


attrition_df.isnull().sum()


# In[ ]:


# attrition_df.columns = attrition_df.columns.str.lower()
attrition_df


# In[ ]:


#CONFIRM NULL VALUES
attrition_df.isnull().values.any().sum()
print("Missing Values, Detail:", '\n', attrition_df.isnull().sum())
print('Total Missing Values:', attrition_df.isnull().sum().sum())


# In[ ]:


attrition_df.columns


# In[ ]:


attrition_df.dtypes


# In[ ]:


attrition_df.shape


# In[ ]:


attrition_df.values


# In[ ]:


attrition_df.info()


# In[ ]:


attrition_df['Education']


# In[ ]:


attrition_df.head()


# In[ ]:


attrition_df.Education.describe()


# In[ ]:


#prints the mean of non categorical data
for column in attrition_df:
    if not column in [pre_categorized_data + post_categorical_data]:
        pp(column)
        pp(attrition_df[column].mean())


# In[ ]:


attrition_df.BusinessTravel.unique()


# In[ ]:


#confirms that data types are no longer strings
attrition_df.applymap(type).eq(str).all()


# In[ ]:


attrition_df.DailyRate


# In[ ]:


attrition_df.head(1)

list(attrition_df.columns)


# In[ ]:


attrition_df.head()


# # Hypothesis

# ### I predict that
# 
# - age
# - education
# - environmentsatisfaction
# - monthlyrate
# - hourlyrate
# - dailyrate
# 
# will be factors/features that will be predictive in finding employee attrition

# ## Visualization

# In[ ]:


hypothesized_predictors = [
"Age",
"Education",
"EnvironmentSatisfaction",
"MonthlyRate",
"HourlyRate",
"DailyRate"
]

# EnvironmentSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
data_env_sat = {'Low': 1, 'Medium': 2, 'High': 3, 'Very High': 4}
data_env_sat_names = list(data_env_sat.keys())
data_env_sat_values = list(data_env_sat.values())

# Education
# 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
data_edu = {'Below College': 1, 'College': 2, 'Bachelor': 3, 'Master': 4, 'Doctor': 5}
data_edu_names = list(data_edu.keys())
data_edu_values = list(data_edu.values())


# In[ ]:


sns.regplot(y=attrition_df['Attrition'], x=attrition_df["DailyRate"], data=attrition_df)


# In[ ]:


# # sns.set_style('ticks')

# sns.regplot(x=attrition_df['Attrition'], y=attrition_df["DailyRate"], data=attrition_df)

# # sns.set_style('ticks')
# # fig, ax = plt.subplots()
# # fig.set_size_inches(18.5, 10.5)
# # sns.regplot(data[:,0], data[:,1], ax=ax)
# # sns.despine()


# In[ ]:


for factor in hypothesized_predictors:
    print(factor)
    if factor == "Education":
        fig, axs = plt.subplots(1, 3, figsize=(9,3), sharey=True)

        axs[0].bar(data_env_sat_names, data_env_sat_values)
        for xtick in axs[0].get_xticklabels():
            xtick.set_rotation(45)
        
        axs[1].scatter(data_env_sat_names, data_env_sat_values)
        for xtick in axs[1].get_xticklabels():
            xtick.set_rotation(45)
        
        axs[2].plot(data_env_sat_names, data_env_sat_values)
        for xtick in axs[2].get_xticklabels():
            xtick.set_rotation(45)
        
        fig.suptitle('Categorical Plotting of Education')

    elif factor == "EnvironmentSatisfaction":
        fig, axs = plt.subplots(1, 3, figsize=(9,3), sharey=True)
        
        axs[0].bar(data_edu_names, data_edu_values)
        for xtick in axs[0].get_xticklabels():
            xtick.set_rotation(45)
        
        axs[1].scatter(data_edu_names, data_edu_values)
        for xtick in axs[1].get_xticklabels():
            xtick.set_rotation(45)
            
        axs[2].plot(data_edu_names, data_edu_values)
        axs[2].tick_params(axis='x', which='major', pad=115)
        for xtick in axs[2].get_xticklabels():
            xtick.set_rotation(45)


        
        fig.suptitle('Categorical Plotting of Environment Satisfaction')
    else:
        plt.hist((attrition_df[factor]), bins=25, ec='black')
        plt.xlabel(factor)
        plt.ylabel('Count')
        plt.show()


# In[ ]:


plt.hist(attrition_df.Age, bins=25, ec='black')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# ### Models
# 
# 

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()


# In[ ]:


attrition_df.columns


# In[ ]:


#setting features and predictors for models
X = attrition_df.drop(['Attrition', 'AttritionBool', 'Department', 'BusinessTravel', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus'], axis=1)
y = attrition_df.AttritionBool