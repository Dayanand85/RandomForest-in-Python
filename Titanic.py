# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 14:32:20 2022

@author: Dayanand
"""
### importing library

import os
import pandas as pd
import numpy as np
import seaborn as sns

### setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

### changing directory
os.chdir("C:/Users/Dayanand/Desktop/DataScience/dsp1/DataSets")

### loading file
rawDf=pd.read_csv("Titanic_train.csv")
predictionDf=pd.read_csv("Titanic_test.csv")

### let us see insight
rawDf.shape
predictionDf.shape
rawDf.columns
predictionDf.columns

#### Let us divide rawDf into train & test
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

### Let us add Survived column in predictionDf
predictionDf["Survived"]=0

### Let us add source column in all three datasets
trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

### Let us combine all three columns
fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)

##### drop identifer columns 
fullDf.columns
fullDf.shape
fullDf.drop(["PassengerId","Name","Ticket"],axis=1,inplace=True)

# Check NULL Values
fullDf.isna().sum()
# Missing values imputation
# Age,Fare,Cabin,Embarked
# Age
fullDf["Age"].dtypes
tempMed=fullDf.loc[fullDf["Source"]=="Train","Age"].median()
fullDf["Age"].fillna(tempMed,inplace=True)
fullDf["Age"].isna().sum()

# Fare
fullDf["Fare"].dtypes
tempMed=fullDf.loc[fullDf["Source"]=="Train","Fare"].median()
fullDf["Fare"].fillna(tempMed,inplace=True)
fullDf["Fare"].isna().sum()

# Cabin
fullDf["Cabin"].dtype
fullDf["Cabin"].fillna("NotAllocated",inplace=True)
fullDf["Cabin"].isna().sum()

# Embarked
fullDf["Embarked"].dtype
tempMode=fullDf.loc[fullDf["Source"]=="Train","Embarked"].mode()[0]
fullDf["Embarked"].fillna(tempMode,inplace=True)
fullDf["Embarked"].isna().sum()

##### class ratio
fullDf.loc[fullDf["Source"]=="Train","Survived"].value_counts()/trainDf.shape[0]
### 0 - 61% ,1 -38%

#####
## Dummy Variable Creation
fullDf2=pd.get_dummies(fullDf)
fullDf2.shape

### Divide the data into train & test
trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
trainDf.shape
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
testDf.shape
predictionDf=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
predictionDf.shape

#### Divide each data sets into train & test
depVar="Survived"
trainX=trainDf.drop(depVar,axis=1)
trainX.shape
trainY=trainDf[depVar]
trainY.shape
testX=testDf.drop(depVar,axis=1)
testX.shape
testY=testDf[depVar]
testY.shape
predictionX=predictionDf.drop(depVar,axis=1)
predictionX.shape

### Decision Tree Model Building

from sklearn.tree import DecisionTreeClassifier,plot_tree
from matplotlib.pyplot import figure,savefig,close

DT_Model=DecisionTreeClassifier(random_state=2410).fit(trainX,trainY)

#### Model Visualization
figure(figsize=[20,10])
DT_Tree=plot_tree(DT_Model,fontsize=10,feature_names=trainX.columns,filled=True,class_names=["0","1"])

#### 
### Prediction & Validation on TestSet
M1_Predict=DT_Model.predict(testX)

from sklearn.metrics import classification_report

confusion_matrix=pd.crosstab(M1_Predict,testY)
confusion_matrix

print(classification_report(M1_Predict,testY))

#####
### Random Forest
#######

from sklearn.ensemble import RandomForestClassifier
RF_Model1=RandomForestClassifier(random_state=2410).fit(trainX,trainY)

# Prediction on Test Set
Test_Predict1=RF_Model1.predict(testX)

# Confusion Matrix
conf_mat=pd.crosstab(testY,Test_Predict1)
conf_mat
# classification report
from sklearn.metrics import classification_report
print(classification_report(testY,Test_Predict1))

# Prediction on prediction data
Test_Prediction_Data=RF_Model1.predict(predictionX)
Test_Prediction_Data
Gender_Submission=pd.concat([predictionDf["PassengerId"],pd.DataFrame(Test_Prediction_Data)],axis=1)
Gender_Submission.to_csv("gender_submission")
# Vriable importance
RF_Model_Var_Imp=pd.concat([pd.DataFrame(RF_Model1.feature_importances_),pd.DataFrame(trainX.columns)],axis=1)
RF_Model_Var_Imp.columns=["Value","Variable"]
RF_Model_Var_Imp.sort_values(["Value"],ascending=False,inplace=True)

tmpMedian=RF_Model_Var_Imp["Value"].median()
tempDf=RF_Model_Var_Imp[RF_Model_Var_Imp["Value"]>tmpMedian]
tempDf.shape
Var_Imp=list(tempDf["Variable"])

# Model on Important Variable
RF_Model2=RandomForestClassifier(random_state=2410).fit(trainX[Var_Imp],trainY)

# Prediction on test Set
Test_Predict2=RF_Model2.predict(testX[Var_Imp])

# Confusion Matrix
pd.crosstab(Test_Predict2,testY)

# Classification Report
print(classification_report(Test_Predict2,testY))

import seaborn as sns
sns.scatterplot(x="Variable",y="Value",data=RF_Model_Var_Imp)

#### RF model with tuning paprameters
RF_Model3=RandomForestClassifier(random_state=2410,n_estimators=75,max_features=5,min_samples_leaf=10)
RF_Model4=RF_Model3.fit(trainX[Var_Imp],trainY)
Test_Predict4=RF_Model4.predict(testX[Var_Imp])

conf_mat4=pd.crosstab(testY,Test_Predict4)
conf_mat4
print(classification_report(testY,Test_Predict4))

### Automate Grid Search

n_estimators_list=[5,10,25,50,75]
max_features_list=[5,7,9,11]
min_samples_leaf_list=[5,10,20,25,30,50]

from sklearn.model_selection import GridSearchCV
my_param_grid={"n_estimators":n_estimators_list,
          "max_features":max_features_list,
          "min_samples_leaf":min_samples_leaf_list}

Grid_Search_CV=GridSearchCV(estimator=RandomForestClassifier(random_state=2410),
                            param_grid=my_param_grid,
                            scoring="accuracy",cv=3).fit(trainX,trainY)

# Let us store information in data frame
Model_Validation_Df=pd.DataFrame(Grid_Search_CV.cv_results_)

# Based on the selected hyper parameters
RF_Model5=RandomForestClassifier(random_state=2410,n_estimators=5,min_samples_leaf=30,
                                 max_features=11).fit(trainX,trainY)

Test_Predict5=RF_Model5.predict(testX)

conf_mat4=pd.crosstab(testY,Test_Predict5)
print(classification_report(testY,Test_Predict5))

