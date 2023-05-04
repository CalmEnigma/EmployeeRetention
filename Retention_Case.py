# Import libraries
import numpy as np
import pandas as pd
import os as os

# Set directory
directory = "C:\\Users\\calmp\\OneDrive\\0 - Job Applications\\Job Apps\\2022 & 2023 Job Hunt\\0 - Interviews\\QuantSpark\\Case Study"
os.chdir(directory)

# Read excel
df = pd.read_excel("New_Interviewee_Case_Study_Dataset_FINAL__282_29.xlsx")



# Quick look into the data

## Check missing data
#### No imputation because correctly missing. If there was no complaint, one
#### cannot give it a resolution or not, and thus cannot given it a years since
#### resolution.
missing = df.isna().sum()

## Numerical data distributions
desc = df.describe().transpose()
desc

## Check categorical data spellings
### Get categorical variables
df_cat = df.select_dtypes(exclude=[np.number])
cats = pd.Series({col:df_cat[col].unique() for col in df_cat})

## Data seems clean. No data cleaning




# Data engineering

## Functions for num conversions
def complaintnum(text):
    if text == "N":
        return 0
    elif text == "Y":
        return 1
    else:
        return None
    
def travelnum(text):
    if text == "Non-Travel":
        return 0
    elif text == "Travel_Rarely":
        return 1
    else:
        return 2

def incomenum(text):
    if text == "low":
        return 0
    elif text == "medium":
        return 1
    else:
        return 2
    

## Turn certain categories into numericals for correlations?
df["Left"] = df["Left"].apply(lambda l: 0 if l == "No" else 1)
df["complaintresolved"] = df["complaintresolved"].apply(lambda l: complaintnum(l))
df["GenderNum"] = df["Gender"].apply(lambda l: 0 if l == "Male" else 1)

df2 = df.copy()
df2["TravelNum"] = df2["BusinessTravel"].apply(lambda l: travelnum(l))
df2["IncomeNum"] = df2["MonthlyIncome"].apply(lambda l: incomenum(l))


## One hot encoding
df[["Non-Travel", "Travel_Frequently", "Travel_Rarely"]] = pd.get_dummies(df["BusinessTravel"])
df[["Income_high", "Income_low", "Income_medium"]] = pd.get_dummies(df["MonthlyIncome"])




# New data description
desc2 = df.describe().transpose()



# Create high performance subset
dfh = df[df['PerformanceRating']>=4]
dfh2 = df2[df2['PerformanceRating']>=4]

dfs = df2[(df2['Department']=="Sales")]
dfhs = dfh2[(dfh2['Department']=="Sales")]







# Correlation tables
corr_all = df2.corr().reset_index()
corr_h = dfh2.corr().reset_index()

corr_all_s = dfs.corr().reset_index()
corr_h_s = dfhs.corr().reset_index()

## Combine correlations on Left
corr_left = corr_all[['index', 'Left']].merge(corr_h[['index', 'Left']], on = "index", how='left')
corr_left.columns = ["Variable", "All employees", "High performers"]
corr_left['Delta'] = corr_left['High performers']-corr_left['All employees']

corr_left_s = corr_all_s[['index', 'Left']].merge(corr_h_s[['index', 'Left']], on = "index", how='left')
corr_left_s.columns = ["Variable", "All employees", "High performers"]
corr_left_s['Delta'] = corr_left_s['High performers']-corr_left_s['All employees']



## Key areas to focus on in visual analysis
threshold = 0.05
key_factors = corr_left[(abs(corr_left['High performers'])>=threshold) | 
                        (abs(corr_left["All employees"])>=threshold) |
                        (abs(corr_left["Delta"])>=threshold)]

key_factors_s = corr_left_s[(abs(corr_left_s['High performers'])>=threshold) | 
                        (abs(corr_left_s["All employees"])>=threshold) |
                        (abs(corr_left_s["Delta"])>=threshold)]

    





# Get dummies for department
df[["HR_department", "R&D_department", "Sales_department"]] = pd.get_dummies(df["Department"])


# Binning

## Age
#bins = [15, 20, 25, 30, 35, 40, 45, 50, np.inf]
#labels = ["15-20","21-25","26-30","31-35","36-40", "41-45", "46-50", "51+"]
#labels = ["age_" + sub for sub in labels]
#df['Age_bin'] = pd.cut(df['Age'], bins=bins, labels=labels)



# Export df
df.to_csv('Retention_df.csv', index=False)





