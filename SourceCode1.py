# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 20:41:51 2022

@author: Dayanand
"""

# loading library

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",1000)
pd.set_option("display.width",500)

# changing the directory
os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\Imarticus hackathon")

# loading file
rawDf=pd.read_csv("Train.csv")
predictionDf=pd.read_csv("Test.csv")

# Analysis of the data sets
rawDf.head()
rawDf.shape
predictionDf.shape 
rawDf.columns
predictionDf.columns # we do not have output columns
rawDf.info()
# let us add output columns
predictionDf["Global_Sales"]=0
predictionDf.shape

# let us divide the data sets
from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape

# let us add source column in all three data sets
trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

# let us concat all three datasets
fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullDf.shape

# let us drop some columns
fullDf.drop(["Name"],axis=1,inplace=True)
fullDf.shape

# let us check missing values
fullDf.isna().sum()
fullDf.info(verbose=True)
fullDf.describe().T

# Observation-User_Score var data type is object.While it is numeric.It is because of one text value entered in column
# let us change into numeric data types
fullDf["User_Score"].value_counts() # we have tbd repeated 2425 times.I hope it is missing values only
# So let us convert it into nan
fullDf["User_Score"]=np.where(fullDf["User_Score"]=="tbd",np.nan,fullDf["User_Score"])   
fullDf["User_Score"].isnull().sum()
fullDf["User_Score"].unique()
fullDf["User_Score"]=pd.to_numeric(fullDf["User_Score"])
fullDf.dtypes

# Missing value treatment

for i in fullDf.columns:
    if (i!="Global_Sales" and i!="Source"):
        if fullDf[i].dtype=="object":
            tempMode=fullDf.loc[fullDf["Source"]=="Train",i].mode()[0]
            fullDf[i].fillna(tempMode,inplace=True)
        else:
            tempMed=fullDf.loc[fullDf["Source"]=="Train",i].median()
            fullDf[i].fillna(tempMed,inplace=True)

# let us check missing value
fullDf.isna().sum()
# Let us transform some columns

# Developer- Let us convert this column values int

freq=fullDf["Developer"].value_counts().to_dict()
freq
fullDf["Developer"]=fullDf["Developer"].map(freq)

freq1=fullDf["Publisher"].value_counts().to_dict()
fullDf["Publisher"]=fullDf["Publisher"].map(freq1)



## EDA 

fullDf.hist(figsize=(20,20))
for i in fullDf.columns:
    if fullDf[i].dtypes=="object":
        sns.boxplot(data=fullDf,x=fullDf[i],y="Global_Sales")

# correlation

cont_vars=["Year_of_Release","Developer","Publisher","NA_Sales","EU_Sales","JP_Sales","Critic_Score",
           "User_Count","Global_Sales"]


corrDf=fullDf[cont_vars].corr()
corrDf.head()

sns.heatmap(corrDf,cmap="YlOrBr",annot=True)

# dummy varibale creation

fullDf2=pd.get_dummies(fullDf)


# Divide the data sets 

Train=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
Train.shape
Test=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)
Prediction=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1)

# Divide data sets into dep and indep var
trainX=Train.drop(["Global_Sales"],axis=1)
trainY=Train["Global_Sales"]

testX=Test.drop(["Global_Sales"],axis=1)
testY=Test["Global_Sales"]

predictionX=Prediction.drop(["Global_Sales"],axis=1)

#Random Forest Model Building

from sklearn.ensemble import RandomForestRegressor
RM1=RandomForestRegressor(random_state=2410).fit(trainX,trainY)

Test_Predict1=RM1.predict(testX)



RMSE=np.sqrt(np.mean((testY-Test_Predict1)**2))
RMSE#.24

#MAPE
(np.mean(np.abs(((testY-Test_Predict1)/testY))))*100
#6.24

# Prediction on PredictionData sets
OutputDf=pd.DataFrame()
OutputDf["Name"]=predictionDf["Name"]
OutputDf["Global sales"]=RM1.predict(predictionX)
OutputDf.to_csv("sample.csv",index=False)

