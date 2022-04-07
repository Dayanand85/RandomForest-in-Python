# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 20:35:45 2022

@author: Dayanand
"""

# loading library

import pandas as pd
import numpy as np
import os
import seaborn as sns

# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# setting directory

os.chdir("C:/Users/Dayanand/Desktop/DataScience/Internship@YoShoP/Participants_dataset")

# loading library

rawData=pd.read_csv("train.csv")
predictionData=pd.read_csv("test.csv")

# compare both of data sets

rawData.shape
rawData.columns
predictionData.shape
predictionData.columns

# Divide the data into train & test

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawData,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# Add source column in all three data sets

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionData["Source"]="Prediction"

# Combine all three columns

fullDf=pd.concat([trainDf,testDf,predictionData],axis=0)
fullDf.shape

# Remove identifier columns
# Accident_ID,Date,Local_Authority_(District),Local_Authority_(Highway),Time,postcode,country

fullDf.drop(["Accident_ID","Date","Local_Authority_(District)","Local_Authority_(Highway)"
             ,"Time","postcode","country"],axis=1,inplace=True)

# check NULL Values

fullDf.isna().sum()

# Road_Surface_Conditions,Special_Conditions_at_Site-NULL Values

# Road_Surface_Conditions

trainDf["Road_Surface_Conditions"].dtypes
tempMode=fullDf.loc[fullDf["Source"]=="Train","Road_Surface_Conditions"].mode()[0]
fullDf["Road_Surface_Conditions"].fillna(tempMode,inplace=True)
fullDf["Road_Surface_Conditions"].isna().sum()

# Special_Conditions_at_Site

trainDf["Special_Conditions_at_Site"].dtypes
tempMode=fullDf.loc[fullDf["Source"]=="Train","Special_Conditions_at_Site"].mode()[0]
fullDf["Special_Conditions_at_Site"].fillna(tempMode,inplace=True)
fullDf["Special_Conditions_at_Site"].isna().sum()

#####Dummy variable Creation

fullDf.shape
fullDf2=pd.get_dummies(fullDf)
fullDf2.shape

###
## Divide the data sets into train,test & prediction

train=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
train.shape
test=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
test.shape
prediction=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
prediction.shape

## Divide data sets into dep and indep varaiable

trainX=train.drop(["Number_of_Casualties"],axis=1)
trainX.shape
trainY=train["Number_of_Casualties"]
testX=test.drop(["Number_of_Casualties"],axis=1)
testX.shape
testY=test["Number_of_Casualties"]
predictionX=prediction.drop(["Number_of_Casualties"],axis=1)
predictionX.shape

# Model Building

from sklearn.ensemble import RandomForestRegressor
RM1=RandomForestRegressor(random_state=2410).fit(trainX,trainY)

# Prediction

RM_Predict=RM1.predict(testX)

# Model Evaluation

from sklearn.metrics import mean_squared_error
MSE1=mean_squared_error(testY,RM_Predict,squared=True)
MSE1


# Feature selection
RM1.feature_importances_

Var_Importnce=pd.concat([pd.DataFrame(RM1.feature_importances_),pd.DataFrame(trainX.columns)],axis=1)

# Grid Search Parameter

n_estimator_list=[25,50,75,100]
min_samples_list=[250,350,500]
max_features_list=[5,7,9]

from sklearn.model_selection import GridSearchCV

my_param_grid={"n_estimators":n_estimator_list,
               "min_samples_leaf":min_samples_list,
               "max_features":max_features_list}

Grid_Search_Model=GridSearchCV(estimator=RandomForestRegressor(random_state=2410),
                               param_grid=my_param_grid,
                               cv=3,n_jobs=-1).fit(trainX,trainY)

Model_Validation=pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)

# Model based on hyper parameter
RF_Model2=RandomForestRegressor(random_state=2410,
                                n_estimators=100,min_samples_leaf=250,
                                max_features=9).fit(trainX,trainY)

# Prediction
Test_Predict2=RF_Model2.predict(testX)

# Evaluation
from sklearn.metrics import mean_squared_error

MSE2=mean_squared_error(testY,Test_Predict2,squared=True)
MSE2

# Prediction
RM_Prediction1=RF_Model2.predict(predictionX)

Output=pd.DataFrame()
Output["postcode"]=predictionData["postcode"]
Output["No_of_Casualties"]=RM_Prediction1
Output=Output.groupby("postcode")["No_of_Casualties"].mean()
Output.to_csv("Submission2.csv")
