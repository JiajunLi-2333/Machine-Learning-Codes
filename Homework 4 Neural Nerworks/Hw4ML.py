# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 16:21:16 2024

@author: Jiajun Li
"""

#First of all, import all necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import Perceptron
import torch
import random
import torch
from torch import nn, optim
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
from imblearn.under_sampling import RandomUnderSampler
#Now we start to load the dataset
data = pd.read_csv("./diabetes - Copy.csv")

#First of all, extract the target variable and the rest of the variables
y = data.iloc[:,0]

#update the data set

data = data.iloc[:, 1:]

#Now we need to normalize the non-dummy variables
non_dummy_cols = data.columns[(data.min() != 0) | (data.max() != 1)]
scaler = StandardScaler()
data[non_dummy_cols] = scaler.fit_transform(data[non_dummy_cols])

x = data
#Now all the non-dummy variables have been normalized
Label = np.array(['HighBP', 'HighChol', 'BMI', 'Smoker', 'Stroke', 'Myocardial', 'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker',
                  'HasHealthcare', 'NotAbleToAffordDoctor', 'GeneralHealth', 'MentalHealth','PhysicalHealth', 'HardToClimbStairs', 'BiologicalSex',
                  'AgeBracket', 'EducationBracket', 'IncomeBracket', 'Zodiac'])

#For all models, we are going to trian the model based on the training set and test the model on the test set


#Homework Question 1
#Build and train a Perceptron (one input layer, one output layer, no hidden layers and
#no activation functions) to classify diabetes from the rest of the dataset. What is the
#AUC of this model? 

'''
#build and train the model
def Perceptron_model(x, y, random_seed, test_size):
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y,test_size = test_size, random_state = random_seed)
    
    #try to resample the data set so that we can eliminate the imbalanced class
    
    Pecep = Perceptron(tol = 1e-3, max_iter = 100, random_state = 1000 )
    Pecep.fit(x_train, y_train)
    y_score = Pecep.decision_function(x_test)
    auc_score  = roc_auc_score(y_test, y_score)
    fpr, tpr, _ = roc_curve(y_test, y_score)

    return fpr, tpr, auc_score 

#try different test_size and different random_seeds
results = {}

test_size = np.arange(0.1,0.6,0.1)
random_seed = [42, 100, 500, 1000]

for size in test_size:
    for seed in random_seed:
        fpr, tpr, auc_score = Perceptron_model(x,y,seed, size)
        
        key = f"Test size {size:.1f}, Seed {seed}"
    
        results[key] = auc_score
        print(f"The AUC score for test size {size:.1f} and seed {seed} is {auc_score}")
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic Perceptron  {size:.1f} and {seed} ')
        plt.legend(loc="lower right")
        plt.show()

#we calculate the average of all the AUC scores
average_auc_score = sum(results.values())/len(results)
print(f"The average auc score is {average_auc_score}") #0.706 around 0.71
'''


'''
#Question 2 
#Build and train a feedforward neural network with at least one hidden layer to classify
#diabetes from the rest of the dataset. Make sure to try different numbers of hidden
#layers and different activation functions (at a minimum reLU and sigmoid). Doing so:
#How does AUC vary as a function of the number of hidden layers and is it dependent
#on the kind of activation function used (make sure to include “no activation function”
#in your comparison). How does this network perform relative to the Perceptron?
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#due to imbalanced class, first step is to balance the data set
ros = RandomUnderSampler()
x,y = ros.fit_resample(x,y)

x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=999999)

# Convert to PyTorch tensors
x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Dataset and DataLoader setup
train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#we build a fully connected neural network


class FCNN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size=1, activations=None):
        super(FCNN, self).__init__()
        # Define available activation functions, excluding 'none'
        activation_functions = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        
        layers = []
        previous_size = input_size

        # Check if activations are provided, and default to 'none' if not
        if activations is None:
            activations = ['none'] * len(hidden_sizes)  # Default to 'none' if no activations are specified

        # Create hidden layers
        for size, activation in zip(hidden_sizes, activations):
            layers.append(nn.Linear(previous_size, size))  # Always add the linear layer
            if activation != 'none':
                if activation in activation_functions:
                    layers.append(activation_functions[activation])
                else:
                    raise ValueError(f"Unsupported activation function: {activation}")
            # If activation is 'none', do not append any activation layer
            previous_size = size

        # Add the output layer
        layers.append(nn.Linear(previous_size, output_size))

        # Register the sequence of layers as a module
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


def train(epoch, model, optimizer, verbose = True):
    model.train()
    epoch_loss = 0
    losses = []
    criterion = nn.BCEWithLogitsLoss()  # Initialize the loss function
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # forward pass through the model
        output = model(data)
        # Ensure target is the correct shape and type
        target = target.float().unsqueeze(1)  # BCEWithLogitsLoss expects float tensor
        
        # forward pass through the BCEWithLogitsLoss
        loss = criterion(output, target)
        # backward propagation
        loss.backward()
        
        optimizer.step()
        epoch_loss += loss.item()  # accumulate loss
        if batch_idx % 100 == 0:
            losses.append(loss.detach())
            if verbose:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
    return losses


def test(model, verbose=False):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0  # to keep track of total number of samples processed
    criterion = nn.BCEWithLogitsLoss(reduction = 'sum')  # Initialize loss function
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            target = target.float().unsqueeze(1)  # Ensure target is correctly formatted

            output = model(data)
            test_loss += criterion(output, target).item()  # Accumulate the loss
            
            # Calculate the number of correct predictions
            pred = (torch.sigmoid(output) > 0.5).float()  # Convert logits to probabilities and then to binary predictions
            correct += (pred == target).sum().item()
            total += target.size(0)  # Increment total count

        test_loss /= total  # Average the loss over the total number of samples
        accuracy = 100. * correct / total  # Calculate accuracy percentage

        if verbose:
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
                test_loss, correct, total, accuracy))

    return test_loss, accuracy

def calculate_auc(model):
    model.eval()
    all_targets = []
    all_probabilities = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            logits = model(data)  # Obtain logits from the model
            probabilities = torch.sigmoid(logits).cpu().numpy()  # Apply sigmoid and move to CPU
            
            all_probabilities.extend(probabilities.flatten())  # Store probabilities
            all_targets.extend(target.cpu().numpy())  # Store actual labels

    # Calculate AUC score using true labels and probabilities
    auc_score = roc_auc_score(all_targets, all_probabilities)
    fpr, tpr, _ = roc_curve(all_targets, all_probabilities)
    
    return fpr,tpr,auc_score

#Now we start to train this fully connected neural network

input_size = x_train.shape[1]
hidden_layers = [10]
output_size = 1
activation = ['relu']
model_fnn = FCNN(input_size, hidden_layers, output_size, activation)
model_fnn.to(device)
optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5,weight_decay=1e-5)
for epoch in range(0, 80):
    train(epoch, model_fnn,optimizer, verbose=False)
    test(model_fnn,verbose = False)


#now we find the auc score for this model 

fpr,tpr,auc_score = calculate_auc(model_fnn)
print(f"The auc_score for fully connected neural network with hidden layer dimension {hidden_layers} and {activation[0]} activation is {auc_score}")
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic with {len(hidden_layers)} hidden layers and {activation[0]} activation')
plt.legend(loc="lower right")
plt.show()
'''






'''
#home work question 3
#Build and train a “deep” network (at least 2 hidden layers) to classify diabetes from
#the rest of the dataset. Given the nature of this dataset, is there a benefit of using a
#CNN for the classification? 

#To make this neural network deep enough, I will make 10 hidden layers
input_size = x_train.shape[1]
hidden_layers = [10,10,10]
output_size = 1
activation = ['relu','relu','relu']
model_fnn = FCNN(input_size, hidden_layers, output_size, activation)
model_fnn.to(device)
optimizer = optim.SGD(model_fnn.parameters(), lr=0.01, momentum=0.5,weight_decay=1e-5)
for epoch in range(0, 80):
    train(epoch, model_fnn,optimizer, verbose=False)
    test(model_fnn,verbose = False)
fpr,tpr,auc_score = calculate_auc(model_fnn)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % auc_score)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'Receiver Operating Characteristic with {len(hidden_layers)} hidden layers and {activation[0]} activation')
plt.legend(loc="lower right")
plt.show()
print(f"The auc_score for fully connected neural network with hidden layer dimension {hidden_layers} and {activation[0]} activation is {auc_score}")
'''











#Homework question 4
#Build and train a feedforward neural network with one hidden layer to predict BMI
# the rest of the dataset. Use RMSE to assess the accuracy of your model. Does
#the RMSE depend on the activation function used? 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class FCBMI(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation='relu'):
        super(FCBMI, self).__init__()
        
        # A dictionary of activations with an added 'none' for no activation
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'none': None  # None implies no activation function
        }
        
        # Ensure the chosen activation exists in the dictionary, default to None if 'none' is chosen
        activation_layer = activations.get(activation, None)
        
        # Construct the network layers
        layers = [nn.Linear(input_size, hidden_size)]
        if activation_layer is not None:  # Only add the activation layer if it's not None
            layers.append(activation_layer)
        layers.append(nn.Linear(hidden_size, output_size))
        
        # Create the sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

def trainBMI(epoch, model,optimizer, verbose = False):
    model.train()
    losses = []
    epoch_loss = 0
    
    criterion = nn.MSELoss()
    
    for  batch_idx, (data, target) in enumerate(BMItrain_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        if batch_idx % 100 == 0:
            losses.append(loss.detach())
            if verbose :
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(BMItrain_loader.dataset),
                    100. * batch_idx / len(BMItrain_loader), loss.item()))
                print(f'Average Loss Epoch {epoch}: {epoch_loss / len(BMItrain_loader)}')

def testBMI(model, verbose = True):
    model.eval()
    test_loss = 0
    total_samples = 0
    criterion = nn.MSELoss(reduction = 'sum')
    with torch.no_grad():
        for data, target in BMItest_loader:
            # send data to device, where the "device" is either a GPU if it exists or a CPU
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output,target)
            test_loss += loss.item()
            total_samples += data.size(0)
    
    test_loss /= total_samples
    rmse = torch.sqrt(torch.tensor(test_loss))
    
    if verbose == True:
        print(f'\nTest set: Average loss: {test_loss:.4f}, RMSE: {rmse:.4f}')
    return test_loss, rmse
        


#set the new data set
dataBMI = pd.read_csv("./diabetes - Copy.csv")
BMI_target = dataBMI["BMI"].values.reshape(-1,1)

xBMI = dataBMI.drop('BMI',axis = 1).values

scaler = StandardScaler()
xBMI = scaler.fit_transform(xBMI)
BMI_target = scaler.fit_transform(BMI_target)

xBMI_train, xBMI_test, yBMI_train, yBMI_test = model_selection.train_test_split(xBMI,BMI_target, test_size = 0.1, random_state = 114514 )


BMItrain_dataset = TensorDataset(torch.tensor(xBMI_train, dtype=torch.float32), torch.tensor(yBMI_train, dtype=torch.float32))
BMItest_dataset = TensorDataset(torch.tensor(xBMI_test, dtype=torch.float32), torch.tensor(yBMI_test, dtype=torch.float32))

BMItrain_loader = DataLoader(BMItrain_dataset, batch_size=64, shuffle=True)
BMItest_loader = DataLoader(BMItest_dataset, batch_size=64, shuffle=False)

input_size = xBMI_train.shape[1]
'''
hidden_size = 10
output_size = 1
activation = 'none'
model_BMI = FCBMI(input_size, hidden_size, output_size, activation)
model_BMI.to(device)
optimizer = optim.SGD(model_BMI.parameters(), lr=0.01, momentum=0.5, weight_decay=1e-5)
for epoch in range(0, 10):
    trainBMI(epoch, model_BMI,optimizer, verbose=False)
    testBMI(model_BMI,verbose = True)

activation_functions = ['None', 'ReLU', 'Sigmoid']
rmse_values = [0.9266, 0.9060, 0.9086]

# Creating the histogram
plt.figure(figsize=(8, 4))
plt.bar(activation_functions, rmse_values, color=['blue', 'green', 'red'])
plt.xlabel('Activation Function')
plt.ylabel('RMSE')
plt.title('Comparison of RMSE Values for Different Activation Functions')
plt.ylim(0.9, 0.94)  # Setting the y-axis limits for better visualization
plt.show()
'''





#question 5
#I choose to build a fully connected neural network to predict BMI and I am applying dropout to prevent overfitting
class FCBMIModified(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, activation='relu'):
        super(FCBMIModified, self).__init__()
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh()
        }
        if activation not in activations:
            activation = 'relu'
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            activations[activation],
       
            nn.Linear(hidden_size, hidden_size),
            activations[activation],
         
            nn.Linear(hidden_size, hidden_size),
            activations[activation],
            
            nn.Linear(hidden_size, hidden_size),
            activations[activation],
            
       
            nn.Linear(hidden_size, output_size)
            )
    def forward(self,x):
        return self.network(x)
'''
hidden_size = 10
output_size = 1
activation = 'relu'
modelBMIM = FCBMIModified(input_size, hidden_size,output_size,activation)
modelBMIM.to(device)
optimizer = optim.SGD(modelBMIM.parameters(), lr=0.01,weight_decay=0.01) #weight_decay is lambda
for epoch in range(0,30):
    trainBMI(epoch, modelBMIM, optimizer, verbose = False)
    testBMI(modelBMIM, verbose = True)
'''
learning_rate = [0.001,0.001,0.01,0.01]
lambdas = [0.001,0.01,0.001,0.01]
rmse_com = [0.9228,0.9090,0.9044,0.9085]

bar_labels = [f'lr={lr}, lambda={lmbd}' for lr, lmbd in zip(learning_rate, lambdas)]

# Plotting the histogram
plt.figure(figsize=(10, 6))
bars = plt.bar(bar_labels, rmse_com, color='skyblue')

# Adding the RMSE values on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval, round(yval, 4), va='bottom')

# Setting the title and labels
plt.title('Histogram of RMSE Comparisons')
plt.xlabel('Configuration (Learning Rate, Lambda)')
plt.ylabel('RMSE')

# Show the plot
plt.show()












