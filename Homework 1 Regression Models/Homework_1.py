# -*- coding: utf-8 -*-
# First of all we should include all the packages needed

import scipy
import numpy as np
import pickle
import sklearn
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

#first of all, we need to load all the data from the csv 
data = np.genfromtxt('C:/Users/Jiajun Li/OneDrive/Desktop/Hw1 For ML/housingUnits.csv', delimiter=',')
#Now we need to select all the useful information we need
data = data[1:,:]
print(data.shape)

#Then filter out the value to be predicted 
Y = data[:,7]
print(Y.shape)


# now we need the variable 2,3
x_2= data[:,1]
x_3 = data[:,2]
print(x_2.shape)
print(x_3.shape)

#then we need to normalize the variable 2 and 3 
#now there are two options: either normalize by variable 4 or 5
x_4 = data[:,3]
x_5 = data[:,4]

N4_2= x_2/x_4
N4_3 = x_3/x_4
N5_2 = x_2/x_5
N5_3 = x_3/x_5

#function to generate plots
def Drawplot(Label, x, y, figure_num):
    
    #generate the linear regression
    model = LinearRegression()
    model = model.fit(x.reshape(-1,1),y.reshape(-1,1))
    y_hat = model.predict(x.reshape(-1,1))
    r_square = model.score(x.reshape(-1,1),y.reshape(-1,1))
    print(Label + 'R^2 = {:.3f}'.format(r_square))
    #now we draw the linear regression plot
    plt.figure(figure_num)
    plt.scatter(x,y,label = 'observation')
    plt.plot(x,y_hat,'r',label = 'Prediction')
    plt.xlabel(Label)
    plt.ylabel('Median_house_value')
    plt.title(Label + 'R^2 = {:.3f}'.format(r_square))
    plt.legend()
    
    # Visualize: actual vs. predicted income (from model)
    
def VisualPlot(Label, x,y,figure_num):
    
    model = LinearRegression()
    model = model.fit(x.reshape(-1,1),y.reshape(-1,1))
    y_hat = model.predict(x.reshape(-1,1))
    r_square = model.score(x.reshape(-1,1),y.reshape(-1,1))
    
    plt.figure(figure_num)
    plt.plot(y_hat,y, 'o', markersize = .75)
    plt.xlabel('Prediction from model') 
    plt.ylabel('Actual income')  
    plt.title(Label+'R^2 = {:.3f}'.format(r_square))
    
    
    
'''
#code for Q1
#first, we draw the predictor 2, 3,4,5 before standardized 

#Predictor 2 before standardized
Drawplot("Total number of Rooms in a given block", x_2,Y,1)
VisualPlot("Total number of Rooms in a given block", x_2,Y,2)

#Predictor 3 before standardized
Drawplot("Number of bedrooms in a given block", x_3,Y,3)
VisualPlot("Number of bedrooms in a given block", x_3,Y,4)

#Predictor 4 
Drawplot("Population in the block", x_4, Y,5)
VisualPlot("Population in the block", x_4,Y,6)

#Predictor 5
Drawplot("Number of households in the block", x_5, Y, 7)
VisualPlot("Number of households in the block", x_5,Y,8)

#Now we find out is normalized variable 2 and 3 a better option 

#Normalize v2 with v4
Drawplot("Normalized Variable 2 with v4", N4_2,Y, 9)
VisualPlot("Normalized Variable 2 with v4", N4_2,Y, 10)

#normalize v2 with v5
Drawplot("Normalized Variable 2 with v5", N5_2,Y, 11)
VisualPlot("Normalized Variable 2 with v5", N5_2,Y, 12)

#Normalize v3 with v4
Drawplot("Normalized Variable3 with v4", N4_3,Y, 13)
VisualPlot("Normalized Variable 3 with v4", N4_3,Y, 14)

#normalize v3 with v5
Drawplot("Normalized Variable 3 with v5", N5_3,Y, 15)
VisualPlot("Normalized Variable 3 with v5", N5_3,Y, 16)

#According to the rsquare, normalize with population is actually a better predictor
'''
#code for Q2
#Well, I have actually answered this part in Q1
'''
#Normalize v2 with v4
Drawplot("Normalized Variable 2 with v4", N4_2,Y, 9)
VisualPlot("Normalized Variable 2 with v4", N4_2,Y, 10)

 #normalize v2 with v5
Drawplot("Normalized Variable 2 with v5", N5_2,Y, 11)
VisualPlot("Normalized Variable 2 with v5", N5_2,Y, 12)
'''
#code for Q3
'''
Variables = np.array(['Median age of the houses in the block (in years)',
                      'Total number of rooms in a given block normalize by population',
                      'Number of bedrooms in a given block normailize by population',
                      'Population in the block',
                      'Number of households in the block',
                      'Median household income in the block',
                      'Proximity to the ocean',
                      'Median house value in the block (in dollars)',
                      ])

for i in range (7):
    if (i == 1):
        Drawplot(Variables[i],N4_2,Y,17+i)
    elif (i == 2):
        Drawplot(Variables[i],N4_3,Y,17+i)
    else:
        Drawplot(Variables[i],data[:,i],Y,17+i)
''' 


#code for Q4

new_x = data[:,:6]
print(new_x.shape)

#first of all, we look at all predictors without normailized
model = LinearRegression()
model = model.fit(new_x[:,:], Y)

y_hat = model.predict(new_x)

r_square = model.score(new_x[:,:], Y)
print(r_square)

plt.figure(30)
plt.plot(y_hat, Y, 'o', markersize = .75)
plt.xlabel('Prediction from model') 
plt.ylabel('Actual median house value')  
plt.title('Multiple Regression Model R^2 = {:.3f}'.format(r_square))


#then we look at all the predictors with variale 2 and 3 normailzed by populaiton
'''
#First of all, we need to re-arrange the data set 
new_data = data.copy()
new_data[:, 1] = N4_2
new_data[:, 2] = N4_3

new_data = new_data[:,:7]
print(new_data.shape)



#Now we look at the plot and the corresponding r^2
model = LinearRegression()
model = model.fit(new_data[:,:7], Y)

y_hat = model.predict(new_data[:,:7])

r_square = model.score(new_data[:,:7], Y)
print(r_square)

plt.figure(31)
plt.plot(y_hat, Y, 'o', markersize = .75)
plt.xlabel('Prediction from model') 
plt.ylabel('Actual median house value')  
plt.title('Multiple Regression Model R^2 = {:.3f}'.format(r_square))
'''





'''
#code for Q5

#First of all, we find out the colinearity between variable 2 and 3
Co_x, Co_y = N4_2.reshape(-1,1), N4_3.reshape(-1,1)

model = LinearRegression().fit(Co_x, Co_y)

y_hat = model.predict(Co_x)

r_square = model.score(Co_x, Co_y)
print("The R^2 between variable 2 and variable 3 after standardized is R^2 = {: .3f}".format(r_square))

plt.figure(35)
plt.scatter(Co_x,Co_y, label = 'Normalized observation')





Co_x1, Co_y1 = x_4.reshape(-1,1), x_5.reshape(-1,1)

model = LinearRegression().fit(Co_x1, Co_y1)

y_hat = model.predict(Co_x1)

r_square = model.score(Co_x1, Co_y1)
print("The R^2 between variable 2 and variable 3 after standardized is R^2 = {: .3f}".format(r_square))

plt.figure(36)
plt.scatter(Co_x1,Co_y1, label = 'Normalized observation')
plt.legend()
'''













