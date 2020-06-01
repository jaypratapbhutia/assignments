# %%
Problem Statement: #data description
I decided to treat this as a classification problem by creating a new binary
variable affair (did the woman have at least one affair?) and trying to
predict the classification for each woman.
Dataset
The dataset I chose is the affairs dataset that comes with Statsmodels. It
was derived from a survey of women in 1974 by Redbook magazine, in
which married women were asked about their participation in extramarital
affairs. More information about the study is available in a 1978 paper from
the Journal of Political Economy.
Description of Variables
The dataset contains 6366 observations of 9 variables:
rate_marriage: woman's rating of her marriage (1 = very poor, 5 = very good)
age: woman's age
yrs_married: number of years married
children: number of children
religious: woman's rating of how religious she is (1 = not religious, 4 = strongly religious)
educ: level of education (9 = grade school, 12 = high school, 14 =
some college, 16 = college graduate, 17 = some graduate school, 20 = advanced degree)
occupation: woman's occupation (1 = student, 2 = farming/semiskilled/unskilled, 3 = "white collar", 
4 =teacher/nurse/writer/technician/skilled, 5 = managerial/business, 6 =professional with advanced degree)
occupation_husb: husband's occupation (same coding as above)
affairs: time spent in extra-marital affairs


# %%
# Code for loading data and modules:
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression 
from sklearn.model_selection import train_test_split 
from sklearn import metrics 
from sklearn.model_selection import cross_val_score 
dta =sm.datasets.fair.load_pandas().data

# %%
dta.head(5)

# %%
# statistical and graphical analysis
dta.describe()

# %%
# DATA transformation and derivation of new attributes

# %%
dta.isnull().sum()

# %%
# add "affair" column: 1 represents having affairs, 0 represents not 
dta['affair'] = (dta.affairs > 0).astype(int)
y, X = dmatrices('affair ~ rate_marriage + age + yrs_married + children + religious + educ + C(occupation) + C(occupation_husb)', dta, return_type="dataframe")


# %%
dta.head()

# %%
# dummy variable creation for different categories of occupation
X = X.rename(columns =
{'C(occupation)[T.2.0]':'occ_2',
'C(occupation)[T.3.0]':'occ_3',
'C(occupation)[T.4.0]':'occ_4',
'C(occupation)[T.5.0]':'occ_5',
'C(occupation)[T.6.0]':'occ_6',
'C(occupation_husb)[T.2.0]':'occ_husb_2',
'C(occupation_husb)[T.3.0]':'occ_husb_3',
'C(occupation_husb)[T.4.0]':'occ_husb_4',
'C(occupation_husb)[T.5.0]':'occ_husb_5',
'C(occupation_husb)[T.6.0]':'occ_husb_6'})
#y = np.ravel(y)

# %%
X.head()

# %%
print(y)
y = np.ravel(y) # flatten y into a 1-D array for consideration as a response variable.
print(y)

# %%
dta.columns
X.columns.shape

# %%
import seaborn as sns

# %%
# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in dta:
    if plotnumber<=10 :
        ax = plt.subplot(4,3,plotnumber)
        sns.distplot(dta[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()

# %%
"""
most of the columns are categorical data
"""

# %%
# let's see how data is distributed for every column
plt.figure(figsize=(20,25), facecolor='white')
plotnumber = 1

for column in X:
    if plotnumber<=17 :
        ax = plt.subplot(6,3,plotnumber)
        sns.distplot(X[column])
        plt.xlabel(column,fontsize=20)
        #plt.ylabel('Salary',fontsize=20)
    plotnumber+=1
plt.tight_layout()

# %%
# selection of ML algorithm based on EDA
# as can be seen from the data we have to categories a person based on the feed back that whether it belongs to a specific
# class of having afair or not having affair or we have to find the probabilty whether she has an affair
# --so we choose logistic regression as the clasification model for this project

lmlr = LogisticRegression()
lmlr = lmlr.fit(X, y)


# %%
# data standardisation and normalisation

# %%
# creating train test split using optimum parameter
train_x,test_x,train_y,test_y=train_test_split(X,y,test_size=0.3,random_state=1)

# %%
# model training using ML algorithm
lmlr=lmlr.fit(train_x,train_y)

# %%
# calculation of training and test accuracy

# %%
lmlr.score(X, y)

# %%
# 72% seems to be a good prediction

# %%
# hyperparameter tuning to get a better accuracy

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model2 = LogisticRegression()
model2.fit(X_train, y_train)

# %%
# predict class labels for the test set
predicted = model2.predict(X_test)
print(predicted)

# %%
probs = model2.predict_proba(X_test)
print (probs)

# %%
print(metrics.accuracy_score(y_test, predicted))
print(metrics.roc_auc_score(y_test, probs[:, 1]))

# %%
scores = cross_val_score(LogisticRegression(), X, y, scoring='accuracy', cv=10)
print(scores)
print(scores.mean())

# %%
import numpy as np

# %%
test_sample = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 3, 25, 3, 1, 4, 16]).reshape(1, -1)

# %%
model2.predict_proba(test_sample)

# %%
# it good to see it gives a 76% probability of being in a affair seen.