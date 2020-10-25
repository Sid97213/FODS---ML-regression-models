#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 17:10:22 2020

@author: siddhi
"""

from random import seed
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

#Split dataset into training and testing set
def train_test_split(dataframe,split=0.70):
    train_size = int(split * len(dataframe))    #print(train_size)
    test_size = len(dataframe) - train_size     #print(test_size)
    dataframe = dataframe.sample(frac=1)        #shuffle rows of the dataframe     
    train = dataframe[:train_size]              #copy first 70% elements into train
    test = dataframe[-test_size:]               #copy last 30% elements into test
    return train,test

def hypothesis(weight,X):
    return weight*X

def Compute_Cost(X,y,weight):
    N = len(X)
    y1 = hypothesis(weight,X)
    y1 = np.sum(y1,axis=1)
    return sum((y1-y)**2)/(2*N)
    
def Gradient_Descent(X,y,weight,alpha,epoch):
    N = len(X)
    Cost_fn = []
    precision = 0.000001
    i = 0
    while(i<epoch):
        y_predicted = hypothesis(weight,X)
        y_predicted = np.sum(y_predicted,axis=1)
        
        for c in range(0,len(X.columns)):
            weight[c] = weight[c] - alpha*(sum((y_predicted-y)*X.iloc[:,c])/N)
       
        j = Compute_Cost(X,y,weight)
        Cost_fn.append(j)
        if(j<precision):
            break
        i += 1
    
    plt.plot(Cost_fn)
    return Cost_fn , weight

def RMSE(df,wt):
    N = len(df)
    X = df.loc[:,'bias':'children']
    y_obs = df.loc[:,'charges']
    
    yi_hat = X.dot(wt)
    diff = (y_obs - yi_hat)**2
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse

            
def Solve(dataframe):
    train, test = train_test_split(dataframe) 
    X = train.loc[:,'bias':'children']
    Y = train.loc[:,'charges']
    
    weight = np.array([0.0]*len(X.columns))
    learning_rate = 0.9         #0.5             
    epoch = 500                 #1001
    
    Cost, weight = Gradient_Descent(X, Y, weight, learning_rate, epoch)
    print("Cost Values after every 100 epochs: ")
    for i in range(0,len(Cost)):
        if(i%100==0):
            print(Cost[i])
    
    loss_train = RMSE(train,weight)
    loss_test = RMSE(test,weight)
    return weight,loss_train,loss_test
    

if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('/home/siddhi/Documents/4-1/FODS/A2/insurance.txt', sep=",",header=None)
    #dataframe = pd.read_csv('D:/4-1/FODS/insurance.txt', sep=",",header=None)
    dataframe.columns = ["age","bmi","children","charges"]
    dataframe = pd.concat([pd.Series(1,index=dataframe.index,name="bias"),dataframe],axis=1)
    
    #Normalize the input variables by dividing each column by the maximum values of that column
    for column in dataframe:
        max_val = np.max(dataframe[column])
        dataframe[column]=dataframe[column]/max_val
    #print(dataframe[:5])
    
    #initalizing variables
    num_samples = 20
    rows,cols = (num_samples,4)
    weights = [[0]*cols]*rows
    train_data_error = np.zeros(num_samples)
    test_data_error = np.zeros(num_samples)
    
    w0_sum,w1_sum,w2_sum,w3_sum = (0,0,0,0)
    train_error_sum,test_error_sum = (0,0)
    
    #sampling the data 20 times to get more accurate results
    for i in range(0,num_samples):
        print("\nSample",i+1)
        weights[i],train_data_error[i],test_data_error[i] = Solve(dataframe)
        w0_sum += weights[i][0]
        w1_sum += weights[i][1]
        w2_sum += weights[i][2]
        w3_sum += weights[i][3]
        train_error_sum += train_data_error[i]
        test_error_sum += test_data_error[i]
        
    train_mean_err = train_error_sum/num_samples
    test_mean_err = test_error_sum/num_samples
    var_train_error = np.var(train_data_error)
    var_test_error = np.var(test_data_error)
    min_train_error = np.amin(train_data_error)
    min_test_error = np.amin(test_data_error)

    print("\nThe regression line is of the form:\n insurance = w0 + w1*age + w2*bmi + w3*children")    
    result = [w0_sum/num_samples, w1_sum/num_samples, w2_sum/num_samples, w3_sum/num_samples]
    print("\nWeights of the independent variables are: ")
    print("w0 = ",result[0])
    print("w1 = ",result[1])
    print("w2 = ",result[2])
    print("w3 = ",result[3])
    
    print("\nMean accuracy prediction of training data: ",train_mean_err)
    print("Variance of accuracy of prediction of training data: ",var_train_error)
    print("Minimum error of training data: ",min_train_error)
    
    print("\nMean accuracy prediction of testing data: ",test_mean_err)
    print("Variance of accuracy of prediction of testing data: ",var_test_error)
    print("Minimum error of testing data: ",min_test_error)
    
    

    
    