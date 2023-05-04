# Import libraries
import numpy as np
import pandas as pd
import os as os
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_white
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Set directory
directory = "C:\\Users\\calmp\\OneDrive\\0 - Job Applications\\Job Apps\\2022 & 2023 Job Hunt\\0 - Interviews\\QuantSpark\\Case Study"
os.chdir(directory)

# Read excel
df = pd.read_csv("Retention_df.csv")


# Drop useless columns
df = df.drop(["Over18", "StandardHours", "Gender", "MonthlyIncome",
              "Department", "BusinessTravel"], axis=1)


# Subset high performers
df = df[df['PerformanceRating']>=4]




# Adjust distributions




# Target split
X = df.drop(['Left'], axis=1)
y = df['Left']







# STATISTICAL INFERENCE


# Drop one dummy per group to avoid multicollinearity
X = X.drop(['Non-Travel', 'HR_department', "Income_medium"], axis=1)

# Create interactions
X['Distance_x_WFH'] = X['DistanceFromHome']*X['workingfromhome']
X['LowIncome_x_Hike'] = X['Income_low']*X['PercentSalaryHike']


# Select main indepedent variables
vars_ = ['DistanceFromHome', 'workingfromhome',
         'Income_low', 'Travel_Frequently', 'Age',
         'Sales_department']



# Select controls
## Added in other variables, but no signifiance
# vars_.append('YearsSinceLastPromotion')


# Retain desired vars
X1 = X[vars_]

# Test for multicollinearity with VIF
## NumCompanies, Total working years = scale with age
vif_data = pd.DataFrame()
vif_data["feature"] = X1.columns
vif_data["VIF"] = [variance_inflation_factor(X1.values, i) for i in range(len(X1.columns))]


# Dropped for multicollineatiy:
    ## JobSat -> separate regression?
    ## SalaryHike
    ## = these factors are represented by underlying variables


# Add interactions
vars_.append('Distance_x_WFH')
#vars_.append('LowIncome_x_Hike')


# Run logit regression
logit_model=sm.Logit(y,X[vars_])
log_result=logit_model.fit(cov_type="HC0")
print(log_result.summary2())


# Just satisfaction
logit_model=sm.Logit(y,X['JobSatisfaction'])
log_result=logit_model.fit(cov_type="HC0")
print(log_result.summary2())









# Subset model for those who made complaints
to_drop = df['complaintfiled'] == 1
Xc = X[to_drop]
yc = y[to_drop]

# Add variable
vars_.remove('Distance_x_WFH')
vars_.append('complaintresolved')

# Run logit regression
logit_model=sm.Logit(yc,Xc[vars_])
log_result_c=logit_model.fit()
print(log_result_c.summary2())











# What determines job satisfaction?

# Re-split
X_sat = df.drop(['JobSatisfaction'], axis=1)
y_sat = df['JobSatisfaction']

# Drop nans 
X_sat = X_sat.drop(['complaintresolved', 'complaintyears'], axis=1)

# Drop other unwatned vars
X_sat = X_sat.drop(['Left', "Travel_Rarely", "HR_department", "Income_medium"], axis=1)

# Drop due to multicol
X_sat = X_sat.drop(['PerformanceRating'], axis=1)

# Recursive feature elimination
from sklearn.feature_selection import RFE
ols = LinearRegression()
rfe = RFE(ols, 10)
rfe = rfe.fit(X_sat, y_sat)
vars_ = list(X_sat.columns[rfe.support_])

# Check multicol
vif_data = pd.DataFrame()
vif_data["feature"] = X_sat[vars_].columns
vif_data["VIF"] = [variance_inflation_factor(X_sat[vars_].values, i) for i in range(len(X_sat[vars_].columns))]

# Run logit regression
ols_sat=sm.OLS(y_sat,X_sat[vars_])
sat_result=ols_sat.fit()
print(sat_result.summary2())

# Run logit only with salary hike
ols_sat=sm.OLS(y_sat,X_sat['PercentSalaryHike'])
sat_result=ols_sat.fit()
print(sat_result.summary2())



# What determines salary hike?

# Re-split
X_sat = df.drop(['PercentSalaryHike'], axis=1)
y_sat = df['PercentSalaryHike']

# Drop nans 
X_sat = X_sat.drop(['complaintresolved', 'complaintyears'], axis=1)

# Drop other unwatned vars
X_sat = X_sat.drop(['Left', "Non-Travel", "HR_department",
                    "Income_medium", 'JobSatisfaction'], axis=1)

# Drop due to multicol
#X_sat = X_sat.drop(['PerformanceRating'], axis=1)

# Recursive feature elimination
from sklearn.feature_selection import RFE
ols = LinearRegression()
rfe = RFE(ols, 11)
rfe = rfe.fit(X_sat, y_sat)
vars_ = list(X_sat.columns[rfe.support_])
#vars_.append('Age')

# Check multicol
vif_data = pd.DataFrame()
vif_data["feature"] = X_sat[vars_].columns
vif_data["VIF"] = [variance_inflation_factor(X_sat[vars_].values, i) for i in range(len(X_sat[vars_].columns))]

# Run logit regression
ols_sat=sm.OLS(y_sat,X_sat[vars_])
sat_result=ols_sat.fit()
print(sat_result.summary2())








# What determines performance rating?

# Re-split
X_sat = df.drop(['PerformanceRating'], axis=1)
y_sat = df['PerformanceRating']

# Drop nans 
X_sat = X_sat.drop(['complaintresolved', 'complaintyears'], axis=1)

# Drop other unwatned vars
X_sat = X_sat.drop(['Left', "Non-Travel", "HR_department",
                    "Income_medium", 'JobSatisfaction', 'PercentSalaryHike',
                    "R&D_department", 'Sales_department'], axis=1)

# Drop due to multicol
#X_sat = X_sat.drop(['PerformanceRating'], axis=1)

# Recursive feature elimination
from sklearn.feature_selection import RFE
ols = LinearRegression()
rfe = RFE(ols, 15)
rfe = rfe.fit(X_sat, y_sat)
vars_ = list(X_sat.columns[rfe.support_])
#vars_.append('Age')

# Check multicol
vif_data = pd.DataFrame()
vif_data["feature"] = X_sat[vars_].columns
vif_data["VIF"] = [variance_inflation_factor(X_sat[vars_].values, i) for i in range(len(X_sat[vars_].columns))]

# Run logit regression
ols_sat=sm.OLS(y_sat,X_sat[vars_])
sat_result=ols_sat.fit()
print(sat_result.summary2())



















# PREDICTION


# Fix predictors
X = X.drop(["complaintresolved", "complaintyears", 'Distance_x_WFH', 'LowIncome_x_Hike'], axis=1)

# Set cross validation
k_folds = KFold(n_splits = 8)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature scaling
scaler = preprocessing.StandardScaler().fit(X_train)
X_scaled = scaler.transform(X_train)

# PCA
pca = PCA(n_components=5)
pca.fit(X_train)
X_pca_train = pca.transform(X_train)
X_pca_test = pca.transform(X_test)




# Fit and run logistic regression
X_used = X_scaled.copy()
mod = LogisticRegression()
mod.fit(X_used, y_train)
y_pred = mod.predict(X_test)
scores = cross_val_score(mod, X_used, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print('Test Accuracy: {:.2f}'.format(mod.score(X_test, y_test)))

# Confusion matrix
confusion_matrix = confusion_matrix(list(y_test), list(y_pred))
print(confusion_matrix)




# Fit and run naive bayes
X_used = X_train.copy()
mod = GaussianNB()
mod.fit(X_used, y_train)
y_pred = mod.predict(X_test)
scores = cross_val_score(mod, X_used, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print('Test Accuracy: {:.2f}'.format(mod.score(X_test, y_test)))

# Confusion matrix
confusion_matrix = confusion_matrix(list(y_test), list(y_pred))
print(confusion_matrix)



# Fit and run SVM
X_used = X_scaled.copy()
mod = svm.SVC()
mod.fit(X_used, y_train)
y_pred = mod.predict(X_test)
scores = cross_val_score(mod, X_used, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print('Test Accuracy: {:.2f}'.format(mod.score(X_test, y_test)))

# Confusion matrix
confusion_matrix = confusion_matrix(list(y_test), list(y_pred))
print(confusion_matrix)



# Fit and run tree
from sklearn import tree
X_used = X_train.copy()
mod = tree.DecisionTreeClassifier()
mod.fit(X_used, y_train)
y_pred = mod.predict(X_test)
scores = cross_val_score(mod, X_used, y_train, cv = k_folds)

print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print('Test Accuracy: {:.2f}'.format(mod.score(X_test, y_test)))

# Confusion matrix
confusion_matrix = confusion_matrix(list(y_test), list(y_pred))
print(confusion_matrix)