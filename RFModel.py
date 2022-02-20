# -*- coding: utf-8 -*-
"""
Created on Fri Feb 11 13:27:23 2022

@author: Dayanand
"""

# Loading library

import os
import pandas as pd
import numpy as np
import seaborn as sns

# increase the display size

pd.set_option("display.max_rows",500)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",1000)

# change directory

os.chdir("C:\\Users\\Dayanand\\Desktop\\DataScience\\dsp1\\Job-a-thon")

# loading file train & test datasets.Calling test file to prediction dataset

rawDf=pd.read_csv("train_0OECtn8.csv")
predictionDf=pd.read_csv("test_1zqHu22.csv")

rawDf.shape
predictionDf.shape


# Analyze columns

rawDf.columns # we have one more column in prediction then raw.
predictionDf.columns

# engagement_score column is more in prediction dataset.Add this column into predictionDf

predictionDf["engagement_score"]=0
predictionDf.shape

# Divide the rawDf into train & test

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawDf,train_size=0.7,random_state=2410)

trainDf.shape
testDf.shape

# Add source column in train,test & prediction for pro preocessing the datas

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionDf["Source"]="Prediction"

# Combining  all three datasets for data processing

fullDf=pd.concat([trainDf,testDf,predictionDf],axis=0)
fullDf.shape

# Dropping identifier columns which are not of use

fullDf.columns
fullDf.drop(["row_id","user_id","category_id","video_id"],axis=1,inplace=True)
fullDf.shape

#  NULL values detection

fullDf.isna().sum() # No Null values

#Bivariate Analysis Continuous Variables:Scatter plot

corrDf=fullDf[fullDf["Source"]=="Train"].corr() #inference always shold be from Train data 
corrDf.head

sns.heatmap(corrDf,
            xticklabels=corrDf.columns,
            yticklabels=corrDf.columns,
            cmap='YlOrBr')

# Bivariate Analysis Categorical Variables:Boxplot

sns.boxplot(y=trainDf["engagement_score"],x=trainDf["gender"])
# Male has more engagement_score than female
sns.boxplot(y=trainDf["engagement_score"],x=trainDf["profession"])
# other and working_professional has almost same levele of engagement_score

# dummy variable creation

fullDf2=pd.get_dummies(fullDf,drop_first=False)
fullDf2.shape

############################
# Divide the data into Train and Test
############################
# Divide the data into Train and Test based on Source column and 
# make sure you drop the source column

# Step 1: Divide into Train and Testest

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy() 
predictDf=fullDf2[fullDf2["Source_Prediction"]==1].drop(["Source_Train","Source_Test","Source_Prediction"],axis=1).copy()

########################
# Sampling into X and Y
########################

# Divide each dataset into Indep Vars and Dep var

depVar="engagement_score"
trainX=trainDf.drop([depVar],axis=1)
trainY=trainDf[depVar]

testX=testDf.drop([depVar],axis=1)
testY=testDf[depVar]

predictX=predictDf.drop([depVar],axis=1)

trainX.shape
trainY.shape
testX.shape
testY.shape
predictX.shape


# Model Building Using RandomForest

from sklearn.ensemble import RandomForestRegressor

M1_RF=RandomForestRegressor(random_state=2410).fit(trainX,trainY)

# Prediction on TestData Sets

Test_Predict=M1_RF.predict(testX)

# Model Evaluation

from sklearn.metrics import r2_score

R2_Score=r2_score(testY,Test_Predict)
R2_Score #0.31085

# Feature Importances

Var_Importance_Df = pd.concat([pd.DataFrame(M1_RF.feature_importances_),
                               pd.DataFrame(trainX.columns)], axis = 1)

Var_Importance_Df
Var_Importance_Df.columns = ["Value", "Variable_Name"]
Var_Importance_Df.sort_values("Value", ascending = False, inplace = True)
Var_Importance_Df


# Selecting significant variables using feature importance

tempMedian = Var_Importance_Df["Value"].median()
tempDf = Var_Importance_Df[Var_Importance_Df["Value"] > tempMedian]
tempDf.shape
impVars = list(tempDf["Variable_Name"])

# Preparing models on significamt variables selected using feature importances

M1_RF_ImpVar = RandomForestRegressor(random_state = 2410).fit(trainX[impVars], trainY)

# predict

TestImp_Predict=M1_RF_ImpVar.predict(testX[impVars])

# r2 score check

R2_ScoreImpVar=r2_score(testY,TestImp_Predict)
R2_ScoreImpVar # 0.3063

# Using only signivicant variables our r2 score has been lower.
# And as we know that randomforest requires more features.And we have already less no of variables i.e 7 features.
# so we will  go ahead will all features

#################
# To improve model -Selecting Hyper parameters using GridSearchCV
#################

n_estimators_List = [25, 50, 75] # range(25,100,25)
max_features_List = [3, 4, 5]
min_samples_leaf_List = [5, 10, 25, 50]


from sklearn.model_selection import GridSearchCV

my_param_grid = {'n_estimators': n_estimators_List, 
                 'max_features': max_features_List, 
                 'min_samples_leaf' : min_samples_leaf_List} 

Grid_Search_Model = GridSearchCV(estimator = RandomForestRegressor(random_state=2410), 
                     param_grid=my_param_grid,  
                     scoring='r2', 
                     cv=3).fit(trainX, trainY) 

Model_Validation_Df=pd.DataFrame(Grid_Search_Model.cv_results_)


#################
# RF Model with tuning parameters/ hyperparameter tuning
#################

M2_RF2 = RandomForestRegressor(random_state = 2410, n_estimators = 25, 
                               max_features = 5, min_samples_leaf = 500)
M2_RF2 = M2_RF2.fit(trainX, trainY)


Test_Pred = M2_RF2.predict(testX)

R2_Score2=r2_score(testY,Test_Pred)
R2_Score2


# Hyerparameter tuning paramater selection based on GridSearchCV(considering high rank & low std. var. in view)

#n_estimators=75
#max_features=5
#min_samples_leaf=50

# Model Building based on hyperparameter selection

M2_RF3=RandomForestRegressor(n_estimators=75,max_features=5,min_samples_leaf=50,
                             random_state=2410).fit(trainX,trainY)

# Prediction on Test Data Sets

Test_Predict3=M2_RF3.predict(testX)

# Model Evaluation
R2_Score3=r2_score(testY,Test_Predict3)
R2_Score3 # 0.3492

# To improve our R2_Score we can go for advanced sets of data preparations

# Let us visualize our datasets

cols_to_consider=["age","followers","views"] 
sns.pairplot(trainX[cols_to_consider]) 

# Let us standardized the variables

trainXCopy=trainX.copy()
testXCopy=testX.copy()
predictXCopy=predictX.copy()

from sklearn.preprocessing import StandardScaler

train_Sampling=StandardScaler().fit(trainXCopy)
trainXStd=train_Sampling.transform(trainXCopy)
testXStd=train_Sampling.transform(testXCopy)
predictXStd=train_Sampling.transform(predictXCopy)


trainXStd=pd.DataFrame(trainXStd,columns=trainXCopy.columns)
testXStd=pd.DataFrame(testXStd,columns=testXCopy.columns)
predictXStd=pd.DataFrame(predictXStd,columns=predictXCopy.columns)

# Final Model
# Model Building on Standardized data

M4_RF4=RandomForestRegressor(n_estimators=75,max_features=5,min_samples_leaf=50,
                             random_state=2410).fit(trainXStd,trainY)

# Predict on Test Set

Test_predict5=M4_RF4.predict(testXStd)

# Model Evaluation

R5_Score=r2_score(testY,Test_predict5)
R5_Score # 0.3492
# Prediction on PredictionDataSets

SampleSubmission=pd.DataFrame()
SampleSubmission["row_id"]=predictionDf["row_id"]
SampleSubmission["engagement_score"]=M2_RF3.predict(predictXStd)
SampleSubmission.to_csv("SampleSubmission.csv",index=False)


