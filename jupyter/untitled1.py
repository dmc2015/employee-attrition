#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:23:54 2018

@author: don
"""

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

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


data_file_path = '../data/ibm-hr-analytics-attrition-dataset.zip'
encoding = 'utf-8-sig'


def main():
    data = create_data()
    attrition_df = define_dataframe(data)
    print(attrition_df)
    export_csv(attrition_df)

def create_data():
    """
    documentation
    """
    
    data = []
    with zipfile.ZipFile(data_file_path) as zfile:
        for name in zfile.namelist():
            with zfile.open(name) as readfile:
                for line in io.TextIOWrapper(readfile, encoding):
                    data.append(line.replace('\n', '').split(','))
    
    return data



def define_dataframe(data):
    """
    documentation
    """
    
    labels=['Age', 'Attrition', 'BusinessTravel', 'DailyRate', 'Department', 
           'DistanceFromHome', 'Education', 'EducationField', 'EducationField'
           "EmployeeCount","EmployeeNumber","EnvironmentSatisfaction","Gender","HourlyRate","JobInvolvement",
           "JobLevel","JobRole","JobSatisfaction","MaritalStatus","MonthlyIncome","MonthlyRate","NumCompaniesWorked",
           "Over18","OverTime","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours",
           "StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany",
           "YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"
          ]
    
    attrition_df = pd.DataFrame(data, columns=labels)
    attrition_df = attrition_df.drop([0])
    
    return attrition_df    

def export_csv(data):
    data.to_csv('attrition_formatted_data.csv', sep='\t', encoding='utf-8')
    


if __name__ == "__main__": 
    main()

