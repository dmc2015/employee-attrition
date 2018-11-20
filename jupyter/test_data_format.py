#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 10:41:57 2018

@author: don
"""

import io
import zipfile
import pandas as pd
import numpy as np

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


### Data Conversion Pt 0 -> Drop Non Predictive Features


#FIND SERIES WITH NO VARIANCE/SINGLE VALUE -> THESE SERIES DO NOT HOLD PREDICTIVE VALUE AND CAN BE DROPPED
for series in attrition_df.columns:
        if (len(attrition_df[series].unique()) <= 1): 
            print(series, ' Will Be Dropped It has only,' ,' : ', attrition_df[series].unique(), ' Values') 
            attrition_df.drop(columns=[series], inplace=True)   
            
            
## Data Conversion Pt I -> Spreading Out Multiple Data Fields


#Setting BusinessTravel Datafields
attrition_df["TravelRarelyBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Travel_Rarely"]
attrition_df["TravelFrequentlyBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Travel_Frequently"]
attrition_df["NonTravelBool"] = attrition_df.BusinessTravel[attrition_df["BusinessTravel"] == "Non-Travel"]

#Setting Department Datafields
attrition_df["SalesDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Sales"]
attrition_df["ResearchAndDevelopmentDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Research & Development"]
attrition_df["HumanResourcesDepartmentBool"] = attrition_df.Department[attrition_df.Department == "Human Resources"]


# Education
# 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
attrition_df["EducationLevelBelowCollege"] = attrition_df.Education[attrition_df.Education == "1"]
attrition_df["EducationLevelCollege"] = attrition_df.Education[attrition_df.Education == "2"]
attrition_df["EducationLevelBachelor"] = attrition_df.Education[attrition_df.Education == "3"]
attrition_df["EducationLevelMaster"] = attrition_df.Education[attrition_df.Education == "4"]
attrition_df["EducationLevelDoctor"] = attrition_df.Education[attrition_df.Education == "5"]


# EnvironmentSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["EnvironmentSatisfactionLow"] = attrition_df.EnvironmentSatisfaction[attrition_df.EnvironmentSatisfaction == "1"]
attrition_df["EnvironmentSatisfactionMedium"] = attrition_df.EnvironmentSatisfaction[attrition_df.EnvironmentSatisfaction == "2"]
attrition_df["EnvironmentSatisfactionHigh"] = attrition_df.EnvironmentSatisfaction[attrition_df.EnvironmentSatisfaction == "3"]
attrition_df["EnvironmentSatisfactionVeryHigh"] = attrition_df.EnvironmentSatisfaction[attrition_df.EnvironmentSatisfaction == "4"]

# JobInvolvement 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["JobInvolvementLow"] = attrition_df.JobInvolvement[attrition_df.JobInvolvement == "1"]
attrition_df["JobInvolvementMedium"] = attrition_df.JobInvolvement[attrition_df.JobInvolvement == "2"]
attrition_df["JobInvolvementHigh"] = attrition_df.JobInvolvement[attrition_df.JobInvolvement == "3"]
attrition_df["JobInvolvementVeryHigh"] = attrition_df.JobInvolvement[attrition_df.JobInvolvement == "4"]

# JobSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["JobSatisfactionLow"] = attrition_df.JobSatisfaction[attrition_df.JobSatisfaction == "1"]
attrition_df["JobSatisfactionMedium"] = attrition_df.JobSatisfaction[attrition_df.JobSatisfaction == "2"]
attrition_df["JobSatisfactionHigh"] = attrition_df.JobSatisfaction[attrition_df.JobSatisfaction == "3"]
attrition_df["JobSatisfactionVeryHigh"] = attrition_df.JobSatisfaction[attrition_df.JobSatisfaction == "4"]

# PerformanceRating 
# 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'
attrition_df["PerformanceRatingLow"] = attrition_df.PerformanceRating[attrition_df.PerformanceRating == "1"]
attrition_df["PerformanceRatingGood"] = attrition_df.PerformanceRating[attrition_df.PerformanceRating == "2"]
attrition_df["PerformanceRatingExcellent"] = attrition_df.PerformanceRating[attrition_df.PerformanceRating == "3"]
attrition_df["PerformanceRatingOutstanding"] = attrition_df.PerformanceRating[attrition_df.PerformanceRating == "4"]


# RelationshipSatisfaction 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["RelationshipSatisfactionLow"] = attrition_df.RelationshipSatisfaction[attrition_df.RelationshipSatisfaction == "1"]
attrition_df["RelationshipSatisfactionMedium"] = attrition_df.RelationshipSatisfaction[attrition_df.RelationshipSatisfaction == "2"]
attrition_df["RelationshipSatisfactionHigh"] = attrition_df.RelationshipSatisfaction[attrition_df.RelationshipSatisfaction == "3"]
attrition_df["RelationshipSatisfactionVeryHigh"] = attrition_df.RelationshipSatisfaction[attrition_df.RelationshipSatisfaction == "4"]

# WorkLifeBalance
# 1 'Bad' 2 'Good' 3 'Better' 4 'Best'
attrition_df["WorkLifeBalanceBad"] = attrition_df.WorkLifeBalance[attrition_df.WorkLifeBalance == "1"]
attrition_df["WorkLifeBalanceGood"] = attrition_df.WorkLifeBalance[attrition_df.WorkLifeBalance == "2"]
attrition_df["WorkLifeBalanceBetter"] = attrition_df.WorkLifeBalance[attrition_df.WorkLifeBalance == "3"]
attrition_df["WorkLifeBalanceBest"] = attrition_df.WorkLifeBalance[attrition_df.WorkLifeBalance == "4"]


#Setting EducationField Datafields
attrition_df["LifeScienceEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Life Sciences']
attrition_df["OtherEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Other']
attrition_df["MedicalEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Medical']
attrition_df["MarketingEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Marketing']
attrition_df["TechnicalEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Technical Degree']
attrition_df["HumanResourcesEducationBool"] = attrition_df.EducationField[attrition_df.EducationField == 'Human Resources']

#Setting Gender Datafields
attrition_df["Male"] = attrition_df.Gender[attrition_df.Gender == 'Male']
attrition_df["Female"] = attrition_df.Gender[attrition_df.Gender == 'Female']


#Setting JobRole Datafields
attrition_df["JobRoleSalesExecutiveBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Executive']
attrition_df["JobRoleResearchScientistBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Scientist']
attrition_df["JobRoleLaboratoryTechnicianBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Laboratory Technician']
attrition_df["JobRoleManufacturingDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manufacturing Director']
attrition_df["JobRoleHealthcareRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Healthcare Representative']
attrition_df["JobRoleManagerBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Manager']
attrition_df["JobRoleSalesRepresentativeBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Sales Representative']
attrition_df["JobRoleResearchDirectorBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Research Director']
attrition_df["JobRoleHumanResourcesBool"] = attrition_df.JobRole[attrition_df.JobRole == 'Human Resources']

#Setting MaritalStatus Datafields
attrition_df["DivorcedBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Divorced']
attrition_df["SingleBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Single']
attrition_df["MarriedBool"] = attrition_df.MaritalStatus[attrition_df.MaritalStatus == 'Married']

#Setting JobLevel Datafields
attrition_df["JobLevel1"] = attrition_df.JobLevel[attrition_df.JobLevel == "1"]
attrition_df["JobLevel2"] = attrition_df.JobLevel[attrition_df.JobLevel == "2"]
attrition_df["JobLevel3"] = attrition_df.JobLevel[attrition_df.JobLevel == "3"]
attrition_df["JobLevel4"] = attrition_df.JobLevel[attrition_df.JobLevel == "4"]
attrition_df["JobLevel5"] = attrition_df.JobLevel[attrition_df.JobLevel == "5"]

#Setting StockOptionLevel Datafields
attrition_df["StockOptionLevel0"] = attrition_df.StockOptionLevel[attrition_df.StockOptionLevel == "0"]
attrition_df["StockOptionLevel1"] = attrition_df.StockOptionLevel[attrition_df.StockOptionLevel == "1"]
attrition_df["StockOptionLevel2"] = attrition_df.StockOptionLevel[attrition_df.StockOptionLevel == "2"]
attrition_df["StockOptionLevel3"] = attrition_df.StockOptionLevel[attrition_df.StockOptionLevel == "3"]
attrition_df["StockOptionLevel4"] = attrition_df.StockOptionLevel[attrition_df.StockOptionLevel == "4"]

# attrition_df["OverTime"] = attrition_df.OverTime.map({"Yes": 1, "No": 0})

# # attrition_df["Over18"] = attrition_df.Over18.map({"Y": 1, "N": 0})

# # attrition_df

# # # attrition_df.Education


# # # for cat_data in attrition_df[pre_categorized_data + post_categorical_data]:
# # # #     print(cat_data, ' : ',attrition_df[cat_data].mode)
# # #     print(cat_data)
    
# # # attrition_df[pre_categorized_data + post_categorical_data].Education.mode()
# attrition_df


## Mapping Data To Booleans

attrition_df["AttritionBool"] = attrition_df.Attrition.map({"Yes": 1, "No": 0})
attrition_df["OverTimeBool"] = attrition_df.OverTime.map({"Yes": 1, "No": 0})
attrition_df["TravelRarelyBool"] = attrition_df["TravelRarelyBool"].map({"Travel_Rarely": 1, np.nan: 0})
attrition_df["NonTravelBool"] = attrition_df["NonTravelBool"].map({"Non-Travel": 1, np.nan: 0})
attrition_df["TravelFrequentlyBool"] = attrition_df["TravelFrequentlyBool"].map({"Travel_Frequently": 1, np.nan: 0})
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


attrition_df["JobRoleSalesExecutiveBool"] = attrition_df["JobRoleSalesExecutiveBool"].map({"Sales Executive": 1, np.nan: 0})
attrition_df["JobRoleResearchScientistBool"] = attrition_df["JobRoleResearchScientistBool"].map({"Research Scientist": 1, np.nan: 0})
attrition_df["JobRoleLaboratoryTechnicianBool"] = attrition_df["JobRoleLaboratoryTechnicianBool"].map({"Laboratory Technician": 1, np.nan: 0})
attrition_df["JobRoleManufacturingDirectorBool"] = attrition_df["JobRoleManufacturingDirectorBool"].map({"Manufacturing Director": 1, np.nan: 0})
attrition_df["JobRoleHealthcareRepresentativeBool"] = attrition_df["JobRoleHealthcareRepresentativeBool"].map({"Healthcare Representative": 1, np.nan: 0})
attrition_df["JobRoleManagerBool"] = attrition_df["JobRoleManagerBool"].map({"Manager": 1, np.nan: 0})
attrition_df["JobRoleSalesRepresentativeBool"] = attrition_df["JobRoleSalesRepresentativeBool"].map({"Sales Representative": 1, np.nan: 0})
attrition_df["JobRoleResearchDirectorBool"] = attrition_df["JobRoleResearchDirectorBool"].map({"Research Director": 1, np.nan: 0})
attrition_df["JobRoleHumanResourcesBool"] = attrition_df["JobRoleHumanResourcesBool"].map({"Human Resources": 1, np.nan: 0})


attrition_df["DivorcedBool"] = attrition_df["DivorcedBool"].map({"Divorced": 1, np.nan: 0})
attrition_df["SingleBool"] = attrition_df["SingleBool"].map({"Single": 1, np.nan: 0})
attrition_df["MarriedBool"] = attrition_df["MarriedBool"].map({"Married": 1, np.nan: 0})


#################
# Re-Mapping Categorical Data
# Education Mapping
# 1 'Below College' 2 'College' 3 'Bachelor' 4 'Master' 5 'Doctor'
attrition_df["EducationLevelBelowCollege"] = attrition_df["EducationLevelBelowCollege"].map({"1": 1, np.nan: 0})
attrition_df["EducationLevelCollege"] = attrition_df["EducationLevelCollege"].map({"2": 1, np.nan: 0})
attrition_df["EducationLevelBachelor"] = attrition_df["EducationLevelBachelor"].map({"3": 1, np.nan: 0})
attrition_df["EducationLevelMaster"] = attrition_df["EducationLevelMaster"].map({"4": 1, np.nan: 0})
attrition_df["EducationLevelDoctor"] = attrition_df["EducationLevelDoctor"].map({"5": 1, np.nan: 0})



# EnvironmentSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["EnvironmentSatisfactionLow"] = attrition_df["EnvironmentSatisfactionLow"].map({"1": 1, np.nan: 0})
attrition_df["EnvironmentSatisfactionMedium"] = attrition_df["EnvironmentSatisfactionMedium"].map({"2": 1, np.nan: 0})
attrition_df["EnvironmentSatisfactionHigh"] = attrition_df["EnvironmentSatisfactionHigh"].map({"3": 1, np.nan: 0})
attrition_df["EnvironmentSatisfactionVeryHigh"] = attrition_df["EnvironmentSatisfactionVeryHigh"].map({"4": 1, np.nan: 0})


# JobInvolvement 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["JobInvolvementLow"] = attrition_df["JobInvolvementLow"].map({"1": 1, np.nan: 0})
attrition_df["JobInvolvementMedium"] = attrition_df["JobInvolvementMedium"].map({"2": 1, np.nan: 0})
attrition_df["JobInvolvementHigh"] = attrition_df["JobInvolvementHigh"].map({"3": 1, np.nan: 0})
attrition_df["JobInvolvementVeryHigh"] = attrition_df["JobInvolvementVeryHigh"].map({"4": 1, np.nan: 0})

# JobSatisfaction
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["JobSatisfactionLow"] = attrition_df["JobSatisfactionLow"].map({"1": 1, np.nan: 0})
attrition_df["JobSatisfactionMedium"] = attrition_df["JobSatisfactionMedium"].map({"2": 1, np.nan: 0})
attrition_df["JobSatisfactionHigh"] = attrition_df["JobSatisfactionHigh"].map({"3": 1, np.nan: 0})
attrition_df["JobSatisfactionVeryHigh"] = attrition_df["JobSatisfactionVeryHigh"].map({"4": 1, np.nan: 0})

# PerformanceRating 
# 1 'Low' 2 'Good' 3 'Excellent' 4 'Outstanding'
attrition_df["PerformanceRatingLow"] = attrition_df["PerformanceRatingLow"].map({"1": 1, np.nan: 0})
attrition_df["PerformanceRatingGood"] = attrition_df["PerformanceRatingGood"].map({"2": 1, np.nan: 0})
attrition_df["PerformanceRatingExcellent"] = attrition_df["PerformanceRatingExcellent"].map({"3": 1, np.nan: 0})
attrition_df["PerformanceRatingOutstanding"] = attrition_df["PerformanceRatingOutstanding"].map({"4": 1, np.nan: 0})


# RelationshipSatisfaction 
# 1 'Low' 2 'Medium' 3 'High' 4 'Very High'
attrition_df["RelationshipSatisfactionLow"] = attrition_df["RelationshipSatisfactionLow"].map({"1": 1, np.nan: 0})
attrition_df["RelationshipSatisfactionMedium"] = attrition_df["RelationshipSatisfactionMedium"].map({"2": 1, np.nan: 0})
attrition_df["RelationshipSatisfactionHigh"] = attrition_df["RelationshipSatisfactionHigh"].map({"3": 1, np.nan: 0})
attrition_df["RelationshipSatisfactionVeryHigh"] = attrition_df["RelationshipSatisfactionVeryHigh"].map({"4": 1, np.nan: 0})


# WorkLifeBalance
# 1 'Bad' 2 'Good' 3 'Better' 4 'Best'
attrition_df["WorkLifeBalanceBad"] = attrition_df["WorkLifeBalanceBad"].map({"1": 1, np.nan: 0})
attrition_df["WorkLifeBalanceGood"] = attrition_df["WorkLifeBalanceGood"].map({"2": 1, np.nan: 0})
attrition_df["WorkLifeBalanceBetter"] = attrition_df["WorkLifeBalanceBetter"].map({"3": 1, np.nan: 0})
attrition_df["WorkLifeBalanceBest"] = attrition_df["WorkLifeBalanceBest"].map({"4": 1, np.nan: 0})

#JOBLEVEL,STOCKOPTION SERIES IS NOT BEING HANDLED PROPERLY, NEEDS RELABELING AND MAPPING

attrition_df["JobLevel1"] = attrition_df["JobLevel1"].map({"1": 1, np.nan: 0})
attrition_df["JobLevel2"] = attrition_df["JobLevel2"].map({"2": 1, np.nan: 0})
attrition_df["JobLevel3"] = attrition_df["JobLevel3"].map({"3": 1, np.nan: 0})
attrition_df["JobLevel4"] = attrition_df["JobLevel4"].map({"4": 1, np.nan: 0})
attrition_df["JobLevel5"] = attrition_df["JobLevel5"].map({"5": 1, np.nan: 0})


attrition_df["StockOptionLevel0"] = attrition_df["StockOptionLevel0"].map({"0": 1, np.nan: 0})
attrition_df["StockOptionLevel1"] = attrition_df["StockOptionLevel1"].map({"1": 1, np.nan: 0})
attrition_df["StockOptionLevel2"] = attrition_df["StockOptionLevel2"].map({"2": 1, np.nan: 0})
attrition_df["StockOptionLevel3"] = attrition_df["StockOptionLevel3"].map({"3": 1, np.nan: 0})
attrition_df["StockOptionLevel4"] = attrition_df["StockOptionLevel4"].map({"4": 1, np.nan: 0})


#confirms that data types are no longer strings
# print(attrition_df.applymap(type).eq(str).all())

#Convert all string numbers to integers
list_of_string_series = ['Age','DailyRate', 'DistanceFromHome',
                         'EmployeeNumber', 'HourlyRate', 'MonthlyIncome',
                         'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike',
                         'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany',
                         'YearsInCurrentRole', 'YearsSinceLastPromotion','YearsWithCurrManager'
                        ]

for series in list_of_string_series:
    attrition_df[series] = pd.to_numeric(attrition_df[series], errors='coerce')
    
    
#drop all columns that are no longer strings
for series in attrition_df.columns:
    if attrition_df[series].dtype !=  np.dtype('int64'):#  attrition_df[series].dtype != np.dtype('O'):
        attrition_df.drop([series], axis=1, inplace=True)


attrition_df.to_csv('attrition.csv', sep='\t')





#assert
