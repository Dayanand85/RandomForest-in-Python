# -*- coding: utf-8 -*-
"""
Created on Tue Jan 18 13:15:05 2022

@author: Dayanand
"""
# loading library

import pandas as pd
import os
import numpy as np
import seaborn as sns

# setting display size

pd.set_option("display.max_rows",1000)
pd.set_option("display.max_columns",500)
pd.set_option("display.width",500)

# setting directory

os.getcwd()
os.chdir("C:\\Users\\tk\\Desktop\\DataScience\\Analytics Vidhya\\DataSet")

# laoding file

rawData=pd.read_csv("HR Analytics\\train_LZdllcl.csv")
predictionData=pd.read_csv("HR Analytics\\test_2umaH9m.csv")

# structure of datasets

rawData.columns
predictionData.columns
rawData.dtypes

# adding is_promoted column in prediction datasets

predictionData["is_promoted"]=0

# sampling of data

from sklearn.model_selection import train_test_split
trainDf,testDf=train_test_split(rawData,train_size=0.7,random_state=2410)
trainDf.shape
testDf.shape


# Adding source column in all three datasets

trainDf["Source"]="Train"
testDf["Source"]="Test"
predictionData["Source"]="Prediction"
trainDf.shape
testDf.shape
predictionData.shape

# combining all three datasets 

fullDf=pd.concat([trainDf,testDf,predictionData],axis=0)
fullDf.shape
fullDf.columns

# removing identifier column employee_id
fullDf.drop(["employee_id"],axis=1,inplace=True)
fullDf.columns
fullDf.shape

# summary of datasets
fullDf.describe

# checking null values
fullDf.isna().sum()

# we have education & previous_year_rating has NULL values
# since we have only 2 columns which have NULL values .So will do manualy

# education-it is an object datatype

fullDf["education"].dtypes
tempMode=fullDf.loc[fullDf["Source"]=="Train","education"].mode()
tempMode
fullDf["education"].fillna(tempMode,inplace=True)
fullDf["education"].isna().sum()

# previous_year ratings
fullDf["previous_year_rating"].dtypes
tempMedian=fullDf.loc[fullDf["Source"]=="Train","previous_year_rating"].median()
tempMedian
fullDf["previous_year_rating"].fillna(tempMedian,inplace=True)
fullDf["previous_year_rating"].isna().sum()

# Outlier analysis

sns.boxplot(y="age",x="is_promoted",data=trainDf)
sns.boxplot(y="avg_training_score",x="is_promoted",data=trainDf)

# Change levels of some column name
# previous_year_rating

fullDf["previous_year_rating"].value_counts()
fullDf["previous_year_rating"].replace({1:"Very Poor",2:"Poor",3:"Average",4:"Good",5:"Very Good"},inplace=True)
fullDf["previous_year_rating"].value_counts()

# KPIs_met>80%
fullDf.dtypes
fullDf["KPIs_met >80%"].value_counts()
fullDf["KPIs_met >80%"].replace({0:"No",1:"Yes"},inplace=True)

# awards_won
fullDf["awards_won?"].value_counts()
fullDf["awards_won?"].replace({0:"No",1:"Yes"},inplace=True)

# Dummy variable creation
fullDf2=pd.get_dummies(fullDf)
fullDf2.shape



# sampling train,test & prediction datasets

trainDf=fullDf2[fullDf2["Source_Train"]==1].drop(["Source_Train","Source_Test"],axis=1)
testDf=fullDf2[fullDf2["Source_Test"]==1].drop(["Source_Train","Source_Test"],axis=1)
predictionDf=fullDf2[(fullDf2["Source_Train"]==0) & (fullDf2["Source_Test"]==0)].drop(["Source_Train","Source_Test"],axis=1)
trainDf.shape
testDf.shape
predictionDf.shape

# divide dependent & independent columns
trainX=trainDf.drop(["is_promoted"],axis=1)
trainY=trainDf["is_promoted"]
testX=testDf.drop(["is_promoted"],axis=1)
testY=testDf["is_promoted"]
predictionX=predictionDf.drop(["is_promoted"],axis=1)
trainX.shape
testX.shape
predictionX.shape
trainY.shape
testY.shape

# model building decision tree

from sklearn.tree import DecisionTreeClassifier,plot_tree
from matplotlib.pyplot import savefig,figure,close

M1=DecisionTreeClassifier(random_state=2410).fit(trainX,trainY)


# Visualization of Tree
figure(figsize=[20,10])
DT_Plot1=plot_tree(M1,feature_names=testX.columns,fontsize=10,filled=True,class_names=["0","1"])

# prediction & validation on Testset
from sklearn.metrics import classification_report

Test_Predict=M1.predict(testX)

# classification Model Validation

conf_mat=pd.crosstab(testY,Test_Predict) #Actual,prediction
conf_mat

# Validation on Testset

print(classification_report(testY,Test_Predict)) #Actual,predict

# Decision Tree bulding with tuning parameters
DT2=DecisionTreeClassifier(random_state=2410,min_samples_leaf=500).fit(trainX,trainY)

# Ploting the tree
figure(figsize=([16,8]))
DT_Plot=plot_tree(DT2,fontsize=10,feature_names=trainX.columns,filled=True,class_names=["0","1"])

# Prediction on Testset
Test_Predict1=DT2.predict(testX)

# Confusion matrix
conf_mat1=pd.crosstab(testY,Test_Predict1)
conf_mat1

# classification report
print(classification_report(testY,Test_Predict1))


# Random Forest
from sklearn.ensemble import RandomForestClassifier
RDF1=RandomForestClassifier(random_state=2410).fit(trainX,trainY)

# Prediction on Testset
Test_Predict2=RDF1.predict(testX)

# Model Validation
conf_mat2=pd.crosstab(testY,Test_Predict2) 

# classification_report
print(classification_report(testY,Test_Predict2))

# features importance
RDF1.feature_importances_

# concat 
feature_imp=pd.concat([pd.DataFrame(RDF1.feature_importances_),pd.DataFrame(trainX.columns)],axis=1)
feature_imp

feature_imp.columns=["Value","Feature_Name"]
feature_imp.sort_values(["Value"],ascending=False,inplace=True)
feature_imp

# use of feature_importance in selecting significant features

tempMedian=feature_imp["Value"].median()
tempMedian
tempDf=feature_imp[feature_imp["Value"]>tempMedian]
tempDf.shape
imp_features=list(tempDf["Feature_Name"])
RF2=RandomForestClassifier(random_state=2410).fit(trainX[imp_features],trainY)

import seaborn as sns
sns.scatterplot(x="Feature_Name",y="Value",data=feature_imp)

# Random forest with tuning parameters

RF_M1=RandomForestClassifier(random_state=2410,n_estimators=25,max_features=5,min_samples_leaf=500)
RF_Model=RF_M1.fit(trainX,trainY)

# prediction on test set
Test_Predict4=RF_Model.predict(testX)

# confusion matrix
conf_mat3=pd.crosstab(testY,Test_Predict4)
conf_mat3


# classification report
print(classification_report(testY,Test_Predict4))

# Manual Grid Searchin

#counter=0

# Tree_List=[]
# Num_Features_List=[]
# Samples_List=[]
# Accuracy_List=[]

# Model_Validation_Df=pd.DataFrame()
# Model_Validation_Df2=pd.DataFrame()
# Model_Validation_Df3=pd.DataFrame()

# for i in n_estimators_list:
#     for j in max_features_list:
#         for k in min_samples_leaf_list:
#             counter=counter+1
#             print(i,j,k)
#             Temp_Model1=RandomForestClassifier(random_state=2410,n_estimatros=i,max_featues=j,min_samples_leaf_list=k)
#             Temp_Model2=Temp_Model1.fit(trainX,trainY)
#             Test_Predict5=Temp_Model2.predict(testX)
#             Temp_Accuracy=sum(np.diag(pd.crosstab(testY,Test_Predict5)))/testX.shape[0]
#             print(i,j,k,Temp_Accuracy)
            
#             Tree_List.append(i)
#             Num_Features_List(j)
#             Samples_List(k)
#             Accuracy_List(Temp_Accuracy)

# Model_Validation_Df=pd.DataFrame({"Trees":Tree_List,"max_features":Num_Features_List,"min_samples_leaf":Samples_List,"Accuracy":Accuracy_List})
            
# GridSearchCV using RandomForest GridSearchCV method

from sklearn.model_selection import GridSearchCV
my_param_grid={"n_estimators":n_estimators_list,"max_features":max_features_list,"min_samples_leaf":min_samples_leaf_list}

Grid_Search_Model=GridSearchCV(estimator=RandomForestClassifier(random_state=2410),param_grid=my_param_grid,scoring="accuracy",cv=3).fit(trainX,trainY)
 
Model_Validation=pd.DataFrame.from_dict(Grid_Search_Model.cv_results_)           
Model_Validation
Model_Validation.to_csv("Model_validation.csv")

# Based on the parameters building model
New_Model=RandomForestClassifier(random_state=2410,n_estimators=25,max_features=5,min_samples_leaf=50).fit(trainX,trainY)

# prediction on test data
Test_Final_Predict=New_Model.predict(testX)

# confusion matrix
conf_final=pd.crosstab(testY,Test_Final_Predict)

# classification report
print(classification_report(testY,Test_Final_Predict))

# prediction on prediction data sets
predictionDf1=M1.predict(predictionX)

HR_analytics_Output1=pd.DataFrame()
HR_analytics_Output1["employee_id"]=predictionData["employee_id"]
HR_analytics_Output1["is_promoted"]=predictionDf1
HR_analytics_Output1.to_csv("HR_analytics_Output.csv",index=False)

