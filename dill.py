# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 14:29:07 2020

@author: Kartikey
"""


#For uploading and accessing the data
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("heart.csv")

df.rename(columns ={'age':'Age','sex':'Sex','cp':'Chest_pain','trestbps':'Resting_blood_pressure','chol':'Cholesterol','fbs':'Fasting_blood_sugar',
                    'restecg':'ECG_results','thalach':'Maximum_heart_rate','exang':'Exercise_induced_angina','oldpeak':'ST_depression','ca':'Major_vessels',
                   'thal':'Thalassemia_types','target':'Heart_attack','slope':'ST_slope'}, inplace = True)

from sklearn.ensemble import RandomForestClassifier

#Libraries for various model parameter selection.
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler


dummy1 = pd.get_dummies(df.Chest_pain)
dummy2 = pd.get_dummies(df.Thalassemia_types)
dummy3 = pd.get_dummies(df.ECG_results)
dummy4 = pd.get_dummies(df.ST_slope)
dummy5 = pd.get_dummies(df.Major_vessels)
merge = pd.concat([df,dummy1,dummy2,dummy3,dummy4,dummy5],axis = 'columns')

final = merge.drop(['Chest_pain','Thalassemia_types','ECG_results','ST_slope','Major_vessels'],axis = 1)

x = final.drop(['Heart_attack'], axis = 1)
y = final['Heart_attack']

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 5)

feature_scaler = MinMaxScaler()
x_train = feature_scaler.fit_transform(x_train)
x_test = feature_scaler.transform(x_test)

accuracy = []



n_estimators = [250,500,750,1000]
criterion = ['gini','entropy']
max_features = ['auto','sqrt','log2']
random_state = [5]

RF = RandomForestClassifier()

parameters = {'n_estimators': [250,500,750,1000],'criterion': ['gini','entropy'],'max_features':['auto','sqrt','log2']}

RFClassifier = GridSearchCV(RF, parameters, scoring='neg_mean_squared_error' ,cv =5)
RFClassifier.fit(x_train, y_train)
RFClassifier.best_params_


model6 = RandomForestClassifier(criterion = 'entropy',max_features = 'log2',n_estimators = 250, random_state = 5)
model6.fit(x_train,y_train)
accuracy6 = model6.score(x_test,y_test)
accuracy.append(accuracy6)
print('Random Forest Classifier Accuracy -->',((accuracy6)*100))

# Predicting the Test set results
y_pred = model6.predict(x_test)
