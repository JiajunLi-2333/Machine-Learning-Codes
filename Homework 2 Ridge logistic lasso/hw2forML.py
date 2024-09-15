import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import model_selection
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import roc_auc_score
from sklearn.metrics import  accuracy_score
from scipy.special import expit
#Load the data
data = pd.read_csv('./techSalaries2017.csv')
data = data.replace({"Title: Senior Software Engineer" : "Male"})

#This is to restrict to the data we wanted
cleaned_data = data.dropna()

cleaned_data = cleaned_data.replace({'Male': 1, 'Female': 0})
cleaned_data = cleaned_data[cleaned_data['gender'] != 'Other']

cleaned_data = cleaned_data.iloc[:,3:]


#this is the output we are trying to get
y = cleaned_data.iloc[:,0]


cdata = cleaned_data.drop(columns = ['totalyearlycompensation','Race', 'Education', 'Some_College', 'Race_Hispanic','basesalary','stockgrantvalue','bonus'])

# Data clean completed



#Code for Question one:
label = np.array(["yearofexperience","yearsatcompeny", "gender", "Master_Degreee",
                  "bachelors", "Doctracte_Degree", "Highshchool_Degree", "Race_Asian",
                  "Race_white", "Race_Two_or_more", "race_black", "Age", "height",
                  "Zodiac", "SAT", "GPA"])

#First of all, we do the multiple regression, including all the predictors
#OLS
x = cdata
'''
#Now we normalize the data
x_norm = (x - np.mean(x, axis=0))/np.std(x, axis=0)
y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0)
model = LinearRegression().fit(x_norm,y_norm.values.reshape(-1,1))
yhat = model.predict(x_norm)
rSqr = model.score(x_norm,y_norm)
RMSE = np.sqrt(metrics.mean_squared_error(y_norm,yhat))
print("The R Squared for multiple regression model is " + str(rSqr) + " RMSE is " + str(RMSE))
#Let us find the betas for further use
betas = model.coef_.flatten()
plt.figure(figsize=(14, 8))
plt.bar(label, betas, color='teal')
plt.xlabel('Features')
plt.ylabel('Beta Coefficients')
plt.title('Beta Coefficients for Each Feature including all predictors')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
'''

#Now each time decrement one feature at one time to see how does it change the result
#Now we are excluding every time, one feature to see the effect on the R^2 and RMSE
'''
for i in range(1,16):
    x = cdata.iloc[:,i:]
    #Again normalize the data
    x_norm = (x - np.mean(x, axis=0))/np.std(x, axis=0)
    y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0)
    model = LinearRegression().fit(x_norm,y_norm.values.reshape(-1,1))
    yhat = model.predict(x_norm)
    rSqr = model.score(x_norm,y_norm)
    RMSE = np.sqrt(metrics.mean_squared_error(y_norm,yhat))
    print(f"The R Squared after removing {label[i-1]} is " + str(rSqr) + " RMSE is " + str(RMSE))
'''

'''
 #Another way of doing this is to do single predictor one by one: 
for i in range(0,16):
    x_single = cdata.iloc[:,i]
    
    #then we normalize the data
    x_norm = (x_single - np.mean(x_single, axis=0))/np.std(x_single, axis=0)
    y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0)
    model = LinearRegression().fit(x_norm.values.reshape(-1,1),y_norm.values.reshape(-1,1))
    yhat = model.predict(x_norm.values.reshape(-1,1))
    rSqr = model.score(x_norm.values.reshape(-1,1),y_norm.values.reshape(-1,1))
    RMSE = np.sqrt(metrics.mean_squared_error(y_norm,yhat))
    print(f"The R Square for {label[i]} is" + str(rSqr) + " RMSE is " + str(RMSE))
    plt.figure() 
    plt.scatter(x_norm, y)
    plt.plot(x_norm,yhat , "b-")
    plt.title(f'Scatter plot of {label[i]} vs Compensation')
    plt.xlabel(label[i])
    plt.ylabel('Compensation')
    plt.show()
'''


#code for q2
'''
#Let us do the ridge regression for all the predictors by using cross-validaiton K-FOLD method
#First we normalize the data
x = cdata
x_norm = (x - np.mean(x, axis=0))/np.std(x, axis=0)
y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0)

#Then set the lambda range and a structure
lambdas = np.linspace(0.0001,1000, 400)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]


# Split the data.Notice that random state is not considered here
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_norm,y_norm, test_size = 0.2,random_state = 10000)


for ii in range(len(lambdas)):
    ridgeModel = Ridge(alpha = lambdas[ii])
    
    split = KFold(5, random_state = 10000,shuffle = True)
    
    cv_scores = cross_val_score(ridgeModel,x_train, y_train,cv = split,scoring = make_scorer(mean_squared_error))
    mean_RMSE = np.mean(np.sqrt(cv_scores))
    
    cont[ii,0] = lambdas[ii]
    cont[ii,1] = mean_RMSE

plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Ridge regression')
plt.show()
print('Optimal lambda:',lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))])


#Build the model with the best lambda

optimal_lambda = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]
ridgeModel = Ridge(alpha = optimal_lambda).fit(x_train,y_train)
RMSE = np.sqrt(mean_squared_error(y_test, ridgeModel.predict(x_test)))
rSqr = ridgeModel.score(x_test,y_test)
Betasridge = ridgeModel.coef_

plt.figure(figsize=(14, 8))
plt.bar(label, Betasridge, color= 'teal')
plt.xlabel('Features')
plt.ylabel("Beta Coefficient for ridge Regression")
plt.title('Beta Coefficients for Each Feature including all predictors')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print("The RMSE for ridge regression is " + str(RMSE)+ " The R^2 for ridgeRegression is " + str(rSqr))


#Now we build a multiple regression model on the same data set to see how does the ridge regression improve
ModelOLS = LinearRegression().fit(x_train,y_train)
BetasOLS = ModelOLS.coef_
plt.figure(figsize=(14, 8))
plt.bar(label, BetasOLS, color = 'red')
plt.xlabel('Features')
plt.ylabel("Beta Coefficient for MultiupleLinear Regression")
plt.title('Beta Coefficients for Each Feature including all predictors')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
'''



#code for q3 Lasso Regression

#first of all, we initialize all the parameters
#Now we are about to perform the lasso regression 



#This time do not normalize the bummy variables
'''
non_binary_columns = ['yearsofexperience', 'yearsatcompany', 'Age', 'Height', 'Zodiac', 'SAT', 'GPA']
cdata[non_binary_columns] = (cdata[non_binary_columns] - np.mean(cdata[non_binary_columns], axis=0)) / np.std(cdata[non_binary_columns], axis=0)
x_norm = cdata
y_norm = (y - np.mean(y, axis=0))/np.std(y, axis=0)
'''
'''

# Split the data.Notice that random state is not considered here
x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.2,random_state = 100000)


#normalize the data
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
y_train_norm = (y_train - np.mean(y_train, axis=0))/np.std(y_train, axis=0)

x_test_norm = (x_test - np.mean(x_test, axis=0))/np.std(x_test, axis=0)
y_test_norm = (y_test - np.mean(y_test, axis=0))/np.std(y_test, axis=0)
#Then set the lambda range and a structure
lambdas = np.linspace(0.0001, 0.05, 200)
cont = np.empty([len(lambdas),2])*np.NaN # [lambda error]


# Split the data.Notice that random state is not considered here




for ii in range(len(lambdas)):
    lassoModel = Lasso(alpha = lambdas[ii])
    
    split = KFold(5,random_state = 100000,shuffle = True) 
    cv_scores = cross_val_score(lassoModel,x_train_norm, y_train_norm,cv = split,scoring = make_scorer(mean_squared_error))
    mean_RMSE = np.mean(np.sqrt(cv_scores))
    
    cont[ii,0] = lambdas[ii]
    cont[ii,1] = mean_RMSE


plt.plot(cont[:,0],cont[:,1])
plt.xlabel('Lambda')
plt.ylabel('RMSE')
plt.title('Lasso regression')
plt.show()
print('Optimal lambda:',lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))])


#Build the model with the best lambda

optimal_lambda = lambdas[np.argmax(cont[:,1]==np.min(cont[:,1]))]

lassoModel = Lasso(alpha = optimal_lambda).fit(x_train_norm, y_train_norm)
RMSE = np.sqrt(mean_squared_error(y_test_norm, lassoModel.predict(x_test_norm)))
rSqr = lassoModel.score(x_test_norm, y_test_norm)


lassoBetas = lassoModel.coef_

plt.figure(figsize=(14, 8))
plt.bar(label, lassoBetas, color= 'teal')
plt.xlabel('Features')
plt.ylabel("Beta Coefficient for lasso Regression")
plt.title('Beta Coefficients for Each Feature including all predictors')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
print("The RMSE for lasso regression is " + str(RMSE) + "The R Squared for lasso regression is " + str(rSqr))
'''
#code for q4 part one 


#redo the data-cleaning process to exlcude only the nan and other in the gender column
sata = data.iloc[:,3:]

sata = sata.replace({'Male': 1, 'Female': 0})
sata = sata[sata['gender'] != 'Other']
sata = sata.dropna(subset = ['gender'])

'''
y_logi = sata['gender'].astype(int) #0 and 1
print(sum(y_logi))
x_logi = sata['totalyearlycompensation'].values.reshape(-1,1)
print(sum(x_logi))


#initialize the logistic regression mode
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')
#split the data set
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_logi,y_logi, test_size = 0.2,random_state = 100000)

#now we should normalize the data:
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

#generate the prediction and the confusion matrix

#fit the model with the train set
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)

print("The ROC_AUC score is " + str(roc_auc))
print("The Accuracy is "  +str(accuracy_score(y_test, logi_predict)))
print('The betas are' + str(logisticRegression.coef_))
#here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()


#now we plot the logistic regression
x_values = np.linspace(x_logi.min(), x_logi.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(x_logi, y_logi, alpha=0.2, c=y_logi, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')
plt.axhline(0.5, color=".5")
# Adding labels and title
plt.xlabel('Total Yearly Compensation')
plt.ylabel('Probability of Being Male')
plt.title('Logistic Regression of Gender on Total Yearly Compensation')
plt.legend()

# Show plot
plt.show()
'''



# Here we include all other factors:
# now further clean the data
sata = sata.drop(columns = ['Race', 'Education'])

logisticRegression = LogisticRegression(class_weight="balanced", solver = 'newton-cg')

y_logi = sata['gender'].astype(int)
x_logi = sata.drop(columns = ['gender'])
x_train, x_test, y_train, y_test = model_selection.train_test_split(x_logi,y_logi, test_size = 0.2,random_state = 100000)

#now we should normalize the data:
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

#generate the prediction and the confusion matrix

#fit the model with the train set
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)

print("The ROC_AUC score is " + str(roc_auc))
print("The Accuracy is "  +str(accuracy_score(y_test, logi_predict)))
print('The betas are' + str(logisticRegression.coef_))
#here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()

#code for q5

#first of all we find the total salary

'''
data = data.iloc[:,3:]

annualSalary = data.iloc[:,0]

#Look for the threshold
salaryMedian = np.median(annualSalary)
print(salaryMedian)

salary_classification = (annualSalary >= salaryMedian).astype(int)

'''

'''
#first we use years of experience to predict
xcol = data.iloc[:,4].values.reshape(-1,1)

    
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')


x_train, x_test, y_train, y_test = model_selection.train_test_split(xcol, salary_classification, test_size = 0.2, random_state = 100000)
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

    
    
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)


print("The ROC_AUC for years of experience is ",  str(roc_auc))
print('The betas are ' + str(logisticRegression.coef_))
print("The Accuracy for years of experience is " + str(accuracy_score(y_test, logi_predict)))
    #here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix HLpay vs years of experience')
plt.show()


    #now we plot the logistic regression
x_values = np.linspace(xcol.min(), xcol.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(xcol,salary_classification, alpha=0.2, c=salary_classification, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')

# Adding labels and title
plt.xlabel('Years Of Experience')
plt.ylabel('Probability')
plt.title('Logistic Regression years of experience  vs HLpay')
plt.legend()

# Show plot
plt.show()
'''


#Now we do for age
'''
xcol = data.iloc[:,19].values.reshape(-1,1)

    
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')


x_train, x_test, y_train, y_test = model_selection.train_test_split(xcol, salary_classification, test_size = 0.2, random_state = 100000)
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

    
    
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)


print("The ROC_AUC for Age is ",  str(roc_auc))
print("The Accuracy for Age is " + str(accuracy_score(y_test, logi_predict)))
print('The betas are ' + str(logisticRegression.coef_))
    #here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix HLpay vs Age')
plt.show()


    #now we plot the logistic regression
x_values = np.linspace(xcol.min(), xcol.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(xcol,salary_classification, alpha=0.2, c=salary_classification, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')

# Adding labels and title
plt.xlabel('Age')
plt.ylabel('Probability')
plt.title('Logistic Regression Age vs HLpay')
plt.legend()

# Show plot
plt.show()
'''
#Now we do for height 
'''
xcol = data['Height'].values.reshape(-1,1)
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')


x_train, x_test, y_train, y_test = model_selection.train_test_split(xcol, salary_classification, test_size = 0.2, random_state = 100000)
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

    
    
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)


print("The ROC_AUC for Height is ",  str(roc_auc))
print("The Accuracy for Hieght is " + str(accuracy_score(y_test, logi_predict)))
print('The betas are ' + str(logisticRegression.coef_))
    #here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix HLpay vs Height')
plt.show()


    #now we plot the logistic regression
x_values = np.linspace(xcol.min(), xcol.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(xcol,salary_classification, alpha=0.2, c=salary_classification, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')

# Adding labels and title
plt.xlabel('Hieght')
plt.ylabel('Probability')
plt.title('Logistic Regression HLpay  vs Height')
plt.legend()

# Show plot
plt.show()
'''
    

#Now we do for SAT
'''
xcol = data['SAT'].values.reshape(-1,1)
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')


x_train, x_test, y_train, y_test = model_selection.train_test_split(xcol, salary_classification, test_size = 0.2, random_state = 100000)
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

    
    
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)


print("The ROC_AUC for SAT is ",  str(roc_auc))
print("The Accuracy for SAT is " + str(accuracy_score(y_test, logi_predict)))
print('The betas are ' + str(logisticRegression.coef_))
    #here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix HLpay vs SAT')
plt.show()


    #now we plot the logistic regression
x_values = np.linspace(xcol.min(), xcol.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(xcol,salary_classification, alpha=0.2, c=salary_classification, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')

# Adding labels and title
plt.xlabel('SAT')
plt.ylabel('Probability')
plt.title('Logistic Regression HLpay  vs SAT')
plt.legend()

# Show plot
plt.show()
'''

#Now we do for GPA
'''
xcol = data['GPA'].values.reshape(-1,1)
logisticRegression = LogisticRegression(class_weight = "balanced", penalty = 'l2',solver = 'newton-cg')


x_train, x_test, y_train, y_test = model_selection.train_test_split(xcol, salary_classification, test_size = 0.2, random_state = 100000)
x_train_norm = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
x_test_norm = (x_test - np.mean(x_test,axis= 0))/np.std(x_test,axis = 0)

    
    
logisticRegression.fit(x_train_norm, y_train)

logi_predict = logisticRegression.predict(x_test_norm)
logi_predict_proba = logisticRegression.predict_proba(x_test_norm)[:,1]
roc_auc = roc_auc_score(y_test, logi_predict_proba)


print("The ROC_AUC for GPA is ",  str(roc_auc))
print("The Accuracy for GPA is " + str(accuracy_score(y_test, logi_predict)))
print('The betas are ' + str(logisticRegression.coef_))
    #here is the confusion matrix we have
confusion_matrix = confusion_matrix(y_test, logi_predict)
plt.figure(figsize=(10, 7))
sns.heatmap(confusion_matrix, annot=True, fmt='g', cmap='Blues', cbar=False,xticklabels=[0, 1], yticklabels=[1, 0])
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix HLpay vs GPA')
plt.show()


    #now we plot the logistic regression
x_values = np.linspace(xcol.min(), xcol.max(), 300).reshape(-1, 1)
x_values_norm = (x_values - np.mean(x_values, axis = 0))/np.std(x_train,axis=0)
probabilities = logisticRegression.predict_proba(x_values_norm)[:, 1]

# Plotting the original data points
plt.scatter(xcol,salary_classification, alpha=0.2, c=salary_classification, cmap='viridis', edgecolors='k')

# Plotting the logistic regression curve
plt.plot(x_values, probabilities, color='red', linewidth=2, label='Logistic Regression Curve')

# Adding labels and title
plt.xlabel('GPA')
plt.ylabel('Probability')
plt.title('Logistic Regression HLpay  vs GPA')
plt.legend()

# Show plot
plt.show()    
    '''
