
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing 
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split


# readng the csv dataset
dat=pd.read_csv('2020_Competition_Training.csv')

# we check the dimension of the imported data
dat.shape

# we display the first 5 rows of the data
dat.head()

# we select the response/dependent variable from dataset
response=dat['transportation_issues']

#predictors=dat.loc[:, ~dat.columns.isin(['transportation_issues','person_id_syn'])]

# extracting the predictors from the entire dataset
pred=dat[['sex_cd','est_age','smoker_current_ind', 'smoker_former_ind','cci_score', 'dcsi_score', 'fci_score','hcc_weighted_sum','hcc_weighted_sum','betos_m5c_pmpm_ct']]

# splitting the data into testing and training set
train_x, test_x, train_y, test_y = train_test_split(pred,response)

#the code below was used to compute correlations among diffent variables
dat.corr()

#selector = SelectKBest(f_classif, k=100)  #chi2
#selector.fit(predictors, response)


# The random forest, logistic regression and decision tree models were built using the code below
forest = RandomForestClassifier()
decision=DecisionTreeClassifier()
logi=LogisticRegression()

# model accuracy was then computed for each machine learning method, and the 
# avearages were calculated

acc_f = cross_val_score(forest, pred,response,cv=5)
acc_f.mean()

acc_d = cross_val_score(decision, pred,response,cv=5)
acc_d.mean()
acc_l = cross_val_score(logi, pred,response,cv=5)

acc_l.mean()

