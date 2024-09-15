# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 13:25:41 2024

@author: Jiajun Li
"""
#import necessary packages
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegressionCV
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import  accuracy_score
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import SVC  
from sklearn import tree
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier
#import data and clean it
data = pd.read_csv('./diabetes.csv')
 
data = data.dropna()
print(data.shape)

#selecting the diabetes
y = data.iloc[:,0]

#Now the rest of the data
data = data.iloc[:, 1:]

#normalize the non-dummy variabels
non_dummy_cols = data.columns[(data.min() != 0) | (data.max() != 1)]
scaler = StandardScaler()
data[non_dummy_cols] = scaler.fit_transform(data[non_dummy_cols])

x = data


#Array of all the predictor 
Label = np.array(['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Myocardial', 'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker',
                  'HasHealthcare', 'NotAbleToAffordDoctor', 'GeneralHealth', 'MentalHealth','PhysicalHealth', 'HardToClimbStairs', 'BiologicalSex',
                  'AgeBracket', 'EducationBracket', 'IncomeBracket', 'Zodiac'])

#Code for Q1: build a logistic Regression model
'''
AUC = 0 
#Balance the class by downsampling the Majority class

for i in range(100):
    rus = RandomUnderSampler(random_state = i)
    x_resampled, y_resampled = rus.fit_resample(x , y)



#split the data
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_resampled ,y_resampled, test_size = 0.3,random_state = 1000)

    model = LogisticRegressionCV(cv= 5, penalty = "l2", scoring = 'roc_auc')

#first of all, build the model with all the predictors
    logiRegression= model.fit(x_train, y_train)


    logi_predict = logiRegression.predict(x_test)
    logi_predict_proba = logiRegression.predict_proba(x_test)[:,1]
    AUC_ROC = roc_auc_score(y_test, logi_predict_proba)
    AUC += AUC_ROC
    fpr, tpr, thresholds = roc_curve(y_test, logi_predict_proba)


AUC = AUC/100
print(f'LogisticRegresssion AUC Score: {AUC}')


differencearray = np.array([])
#Now we need to find the best predictor by droping out each of the the predictors
for i in range(21):
    
    AUC2 = 0

    x_AUC = data.drop(Label[i], axis = 1).copy()
    
    for j in range(10):
        
        rus = RandomUnderSampler(random_state = i)
        x_resampled, y_resampled = rus.fit_resample(x_AUC , y)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_resampled ,y_resampled, test_size = 0.3,random_state = 1000)
        model = LogisticRegressionCV(cv= 5, penalty = "l2", scoring = 'roc_auc', max_iter = 100)
        logiRegression= model.fit(x_train, y_train)
        logi_predict = logiRegression.predict(x_test)
        logi_predict_proba = logiRegression.predict_proba(x_test)[:,1]
        AUC_ROC2 = roc_auc_score(y_test, logi_predict_proba)
        AUC2 += AUC_ROC2
    
    AUC2 = AUC2/10
    print(f'LogisticRegression AUC Score after excluding {Label[i]}: {AUC2}')
    difference = AUC - AUC2
    print(f'LogisticRegression: The AUC drops by: {difference}')
    differencearray = np.append(differencearray, difference)
   


#General Health and BMI should be the best predictor
# Now we plot the bar chart to show the AUC drop
plt.figure(figsize=(10, 6))  # Adjust the size as needed
plt.bar(Label, differencearray, color='skyblue')

# Add labels and title
plt.xlabel('Predictors')
plt.ylabel('AUC Drop')
plt.title('LogisticRegressin: Impact of Excluding Each Predictor on AUC')
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()
'''
















'''
#Code for Question 2
#the class imbalance is indeed an issue so balance the data set 

AUC_SVM = 0
for i in range(100):
    random_seed = i
    rus = RandomUnderSampler(random_state= random_seed)
    x_resampled, y_resampled = rus.fit_resample(x , y)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x_resampled ,y_resampled, test_size = 0.3,random_state = 1000)
    
    
    #tune the hyperparameter
    C_values = np.linspace(0.001, 100, 100)
    param_C = {'C': C_values}
    model = LinearSVC(dual = False, random_state = 42)
    
    grid_search = GridSearchCV(model, param_C, scoring = 'roc_auc', cv = 5)
    grid_search.fit(x_train, y_train)
    
    best_model = grid_search.best_estimator_
    calibrated_svc = CalibratedClassifierCV(best_model, method='sigmoid', cv='prefit')
    calibrated_svc.fit(x_train, y_train)
    y_proba = calibrated_svc.predict_proba(x_test)[:, 1]
    # Calculate the AUC score
    auc_score = roc_auc_score(y_test, y_proba)
    AUC_SVM += auc_score

AUC_SVM /= 100 
print(f'Linear SVM AUC Score: {AUC_SVM}')
#Linear SVM AUC Score: 0.8185631975963753

differenceArraySVM = np.array([])

for i in range(21):
    x1 = data.drop(Label[i], axis = 1).copy()
    AUC_SVM2 = 0
    for j in range(10):
        
        rus1 = RandomUnderSampler(random_state=j)
        x_resampled1, y_resampled1 = rus1.fit_resample(x1 , y)
        x_train1, x_test1, y_train1, y_test1 = model_selection.train_test_split(x_resampled1 ,y_resampled1, test_size = 0.3,random_state = 1000)
        C_values1 = np.linspace(0.001, 100, 10)
        param_C1 = {'C': C_values1}
        
        model1 = LinearSVC(dual = False, random_state = 42)
        grid_search1 = GridSearchCV(model1, param_C1, scoring = 'roc_auc', cv = 5)
        grid_search1.fit(x_train1, y_train1)
        
        best_model1 = grid_search1.best_estimator_
        
        calibrated_svc1 = CalibratedClassifierCV(best_model1, method='sigmoid', cv='prefit')
        calibrated_svc1.fit(x_train1, y_train1)
        y_proba1 = calibrated_svc1.predict_proba(x_test1)[:, 1]
        # Calculate the AUC score
        roc_auc = roc_auc_score(y_test1, y_proba1)
        AUC_SVM2 += roc_auc
    AUC_SVM2 /= 10
    print(f'SVM: AUC Score after excluding {Label[i]}: {AUC_SVM2}')
    difference = AUC_SVM - AUC_SVM2
    print(f'SVM: The AUC drops by: {difference}')
    differenceArraySVM = np.append(differenceArraySVM, difference)

# Now we plot the bar chart to show the AUC drop
plt.figure(figsize=(10, 6))  # Adjust the size as needed
plt.bar(Label, differenceArraySVM, color='skyblue')

# Add labels and title
plt.xlabel('Predictors')
plt.ylabel('AUC Drop')
plt.title('SVM: Impact of Excluding Each Predictor on AUC')
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap

# Show the plot
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()

#The GeneralHealth should be the best predictor
'''
























#code for q3
'''
#first we work on the full predictor
AUC_DecisionTree = 0
differenceArrayDT = np.array([])

param_grid = {
    'criterion': ['gini'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

for i in range(50):
    rusDT = RandomUnderSampler(random_state = i )
    x_resampledDT, y_resampledDT = rusDT.fit_resample(x , y)
    x_trainDT, x_testDT, y_trainDT, y_testDT = model_selection.train_test_split(x_resampledDT ,y_resampledDT, test_size = 0.3,random_state = 1000)
    
    #Now we build the model
    clf = tree.DecisionTreeClassifier()
    grid_search = RandomizedSearchCV(estimator = clf, param_distributions = param_grid, cv = 5, scoring = 'roc_auc')
    grid_search.fit(x_trainDT, y_trainDT)
    
    best_clf = grid_search.best_estimator_
    
    clf_predict_probaDT = best_clf.predict_proba(x_testDT)[:, 1]
    
    roc_aucDT = roc_auc_score(y_testDT, clf_predict_probaDT)
    AUC_DecisionTree += roc_aucDT

AUC_DecisionTree /= 50
print(f'AUC Score for DecisioinTree : {AUC_DecisionTree}')
#AUC Score for DecisioinTree : 0.6511530542925744



#Now we work on excluding each of the predictors
for i in range(21):
    x_DT = data.drop(Label[i], axis = 1).copy()
    AUC_DT = 0
    for j in range(10):
        rusDT2 = RandomUnderSampler(random_state = j )
        x_resampledDT2, y_resampledDT2 = rusDT2.fit_resample(x_DT , y)
        x_trainDT2, x_testDT2, y_trainDT2, y_testDT2 = model_selection.train_test_split(x_resampledDT2 ,y_resampledDT2, test_size = 0.3,random_state = 1000)
        
        #Now we build the model
        clf2 = tree.DecisionTreeClassifier()
        grid_search2 = RandomizedSearchCV(estimator = clf2, param_distributions = param_grid, cv = 5, scoring = 'roc_auc')
        grid_search2.fit(x_trainDT2, y_trainDT2)
        best_clf2 = grid_search2.best_estimator_
        
        clf_predict_probaDT2 = best_clf2.predict_proba(x_testDT2)[:, 1]
        
        roc_aucDT2 = roc_auc_score(y_testDT2, clf_predict_probaDT2)
        AUC_DT += roc_aucDT2
    AUC_DT /= 10
    print(f'DecisionTree: AUC Score after excluding {Label[i]}: {AUC_DT}')
    differenceDT = AUC_DecisionTree - AUC_DT
    print(f'DecisionTree: The AUC drops by: {differenceDT}')
    differenceArrayDT = np.append(differenceArrayDT, AUC_DecisionTree - AUC_DT)


# Now we plot the bar chart to show the AUC drop
plt.figure(figsize=(10, 6))  # Adjust the size as needed
plt.bar(Label, differenceArrayDT, color='skyblue')

# Add labels and title
plt.xlabel('Predictors')
plt.ylabel('AUC Drop')
plt.title('DecisionTree: Impact of Excluding Each Predictor on AUC')
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap
# Show the plot
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()
'''


























#Code for q4
'''
AUC_RF = 0
differenceArrayRF = np.array([])
param_dist = {
    'n_estimators': [10, 50, 100, 200], # Number of trees in the forest
    'max_depth': [None, 10, 20, 30], # Maximum depth of the tree
    'min_samples_split': [2, 5, 10], # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4], # Minimum number of samples required to be at a leaf node
    'bootstrap': [True], # Whether bootstrap samples are used when building trees
    'criterion': ['gini'] # The function to measure the quality of a split
}
for i in range(50):
    rusRF = RandomUnderSampler(random_state = i )
    x_resampledRF, y_resampledRF = rusRF.fit_resample(x , y)
    x_trainRF, x_testRF, y_trainRF, y_testRF = model_selection.train_test_split(x_resampledRF ,y_resampledRF, test_size = 0.3,random_state = 1000)
    
    randomF = RandomForestClassifier(random_state=i)
    
    # Setup RandomizedSearchCV
    rnd_search = RandomizedSearchCV(estimator=randomF, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=i, n_jobs=-1)
    rnd_search.fit(x_trainRF, y_trainRF)
    
    # Use the best estimator
    best_rf = rnd_search.best_estimator_
    
    RF_predict_proba = best_rf.predict_proba(x_testRF)[:, 1]
    roc_aucrf = roc_auc_score(y_testRF, RF_predict_proba)
    
    AUC_RF += roc_aucrf

AUC_RF /= 50
print(f'AUC Score for RandomForest : {AUC_RF}')
#AUC Score for RandomForest : 0.8082206615065414


#Now we shold find the best predictor
for i in range(21):
    x_RF = data.drop(Label[i], axis = 1).copy()
    AUC_RF2 = 0
    
    for j in range(10):
        rusRF2 = RandomUnderSampler(random_state = i )
        x_resampledRF2, y_resampledRF2 = rusRF.fit_resample(x_RF , y)
        x_trainRF2, x_testRF2, y_trainRF2, y_testRF2 = model_selection.train_test_split(x_resampledRF2 ,y_resampledRF2, test_size = 0.3,random_state = 1000)
        
        randomF2 = RandomForestClassifier(random_state=i)
        
        # Setup RandomizedSearchCV
        rnd_search2 = RandomizedSearchCV(estimator=randomF2, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=i, n_jobs=-1)
        rnd_search2.fit(x_trainRF2, y_trainRF2)
        
        # Use the best estimator
        best_rf2 = rnd_search2.best_estimator_
        
        RF_predict_proba2 = best_rf2.predict_proba(x_testRF2)[:, 1]
        roc_aucrf2 = roc_auc_score(y_testRF2, RF_predict_proba2)
        AUC_RF2 += roc_aucrf2
    
    AUC_RF2 /= 10
    print(f'RandomForest: AUC Score after excluding {Label[i]}: {AUC_RF2}')
    differenceRF = AUC_RF - AUC_RF2
    print(f'RandomForest: The AUC drops by: {differenceRF}')
    differenceArrayRF = np.append(differenceArrayRF, differenceRF)
# Now we plot the bar chart to show the AUC drop
plt.figure(figsize=(10, 6))  # Adjust the size as needed
plt.bar(Label, differenceArrayRF, color='skyblue')

# Add labels and title
plt.xlabel('Predictors')
plt.ylabel('AUC Drop')
plt.title('RandomForest: Impact of Excluding Each Predictor on AUC')
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap
# Show the plot
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()
'''

























#Code for q5
AUC_ada = 0
differenceArrayAda = np.array([])
param_dist = {
    'estimator__max_depth': [1, 2, 3, 4, 5],
    'n_estimators': [10, 50, 100, 200],
    'learning_rate': [0.01, 0.1, 1, 10],
    'algorithm': ['SAMME', 'SAMME.R']
}

#first of all, we look for all the predictors
for i in range(50):
    rusAda = RandomUnderSampler(random_state = i )
    x_resampledAda, y_resampledAda = rusAda.fit_resample(x , y)
    x_trainAda, x_testAda, y_trainAda, y_testAda = model_selection.train_test_split(x_resampledAda ,y_resampledAda, test_size = 0.3,random_state = 1000)
    

    base_estimator = DecisionTreeClassifier()
    ada_boost = AdaBoostClassifier(estimator=base_estimator, random_state=i)
    rnd_search_ada = RandomizedSearchCV(estimator=ada_boost, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=i, n_jobs=-1)
    rnd_search_ada.fit(x_trainAda, y_trainAda)
    best_ada = rnd_search_ada.best_estimator_
    ada_predict_proba = best_ada.predict_proba(x_testAda)[:, 1]
    
    AUC_ada += roc_auc_score(y_testAda, ada_predict_proba)
    
AUC_ada /= 50
print(f'AUC Score for adaBoost : {AUC_ada}')
    
    

#then we look for the best predictor
for i in range(21):
    AUC_ada2 = 0
    x_ada = data.drop(Label[i], axis = 1).copy()
    
    for j in range(10):
        rusAda2 = RandomUnderSampler(random_state = i )
        x_resampledAda2, y_resampledAda2 = rusAda.fit_resample(x_ada, y)
        x_trainAda2, x_testAda2, y_trainAda2, y_testAda2 = model_selection.train_test_split(x_resampledAda2 ,y_resampledAda2, test_size = 0.3,random_state = 1000)
        

        base_estimator2 = DecisionTreeClassifier()
        ada_boost2 = AdaBoostClassifier(estimator=base_estimator2, random_state=i)
        rnd_search_ada2 = RandomizedSearchCV(estimator=ada_boost2, param_distributions=param_dist, n_iter=10, cv=5, scoring='roc_auc', random_state=i, n_jobs=-1)
        rnd_search_ada2.fit(x_trainAda2, y_trainAda2)
        best_ada2 = rnd_search_ada2.best_estimator_
        ada_predict_proba2 = best_ada2.predict_proba(x_testAda2)[:, 1]
        
        AUC_ada2 += roc_auc_score(y_testAda2, ada_predict_proba2)
    AUC_ada2 /= 10
    print(f'AdaBoost: AUC Score after excluding {Label[i]}: {AUC_ada2}')
    differenceAda = AUC_ada - AUC_ada2
    print(f'AdaBoost: The AUC drops by: {differenceAda}')
    differenceArrayAda = np.append(differenceArrayAda, differenceAda)
    
    
# Now we plot the bar chart to show the AUC drop
plt.figure(figsize=(10, 6))  # Adjust the size as needed
plt.bar(Label, differenceArrayAda, color='skyblue')

# Add labels and title
plt.xlabel('Predictors')
plt.ylabel('AUC Drop')
plt.title('AdaBoost: Impact of Excluding Each Predictor on AUC')
plt.xticks(rotation=45, ha="right")  # Rotate labels to prevent overlap
# Show the plot
plt.tight_layout()  # Adjust layout to make room for the rotated x-axis labels
plt.show()















