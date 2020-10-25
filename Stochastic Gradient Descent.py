#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 21:11:16 2020

@author: siddhi
"""
from random import seed
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

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

def Compute_Cost(X,y,weight,b,train_data_size):
    y1 = hypothesis(weight,X)+b
    y1 = np.sum(y1,axis=1)
    return sum((y1-y)**2)/(2*train_data_size)


def SGD(train,weight,alpha,epoch,k):
    Cost_fn = []
    b = 0
    curr_iter=1
    
    while(curr_iter<=epoch):
        temp = train.sample(k)          #choosing a random sample of k data points
        Y = np.array(temp['charges'])
        X = np.array(temp.drop('charges',axis=1))
        w_grad = np.zeros(3)
        b_grad = 0
        
        for j in range(0,k):
            y_predicted = np.dot(weight,X[j])
            w_grad = w_grad - 2*X[j]*(Y[j]-y_predicted)
            b_grad = b_grad - 2*(Y[j]-y_predicted)
        
        weight = weight - alpha*(w_grad/k)
        b = b - alpha*(b_grad/k)
        cost = Compute_Cost(X,Y,weight,b,k)
        Cost_fn.append(cost)
        curr_iter += 1
    
    plt.plot(Cost_fn)
    return weight,b,Cost_fn

def RMSE(df,wt,b):
    N = len(df)
    X = df.loc[:,'age':'children']
    y_obs = df.loc[:,'charges']
    
    yi_hat = X.dot(wt)+b
    diff = (y_obs - yi_hat)**2
    rmse = math.sqrt(np.sum(diff)/N)
    return rmse


def Solve(dataframe):
    #Splitting the data into training and testing data
    train, test = train_test_split(dataframe) 
    
    learning_rate = 0.035     #0.0307
    epoch = 1000
    sample_size = 80
    weight = np.zeros(3)
    
    weight,b,Cost = SGD(train,weight,learning_rate,epoch,sample_size)
    print("Cost Values after every 100 epochs: ")
    for i in range(0,len(Cost)):
        if(i%200==0):
            print(Cost[i])
    
    train_loss = RMSE(train,weight,b)
    test_loss = RMSE(test,weight,b)
    return weight,b,train_loss, test_loss
    
            
if __name__ == '__main__':
    seed(1)
    dataframe = pd.read_csv('/home/siddhi/Documents/4-1/FODS/A2/insurance.txt', sep=",",header=None)
    #dataframe = pd.read_csv('D:/4-1/FODS/insurance.txt', sep=",",header=None)
    dataframe.columns = ["age","bmi","children","charges"]
    
    #Normalize the input variables by dividing each column by the maximum values of that column
    for column in dataframe:
        max_val = np.max(dataframe[column])
        dataframe[column]=dataframe[column]/max_val
    #print(dataframe[:5])

    #initializing the variables
    num_samples = 20
    rows,cols = (num_samples,3)
    weights = np.zeros((num_samples,3))
    b_val = np.zeros(num_samples)
    train_data_error = np.zeros(num_samples)
    test_data_error = np.zeros(num_samples)
    
    b_sum,w1_sum,w2_sum,w3_sum = (0,0,0,0)
    err_train_sum,err_test_sum = (0,0)
    
    #Sampling the data 20 times to get more accurate results
    for i in range(0,num_samples):
        print("\nSample ",i+1)
        weights[i],b_val[i],train_data_error[i],test_data_error[i] = Solve(dataframe)
        b_sum += b_val[i]
        w1_sum += weights[i][0]
        w2_sum += weights[i][1]
        w3_sum += weights[i][2]
        err_train_sum += train_data_error[i]
        err_test_sum += test_data_error[i]
        
    train_mean_err = err_train_sum/num_samples
    test_mean_err = err_test_sum/num_samples
    var_train_error = np.var(train_data_error)
    var_test_error = np.var(test_data_error)
    min_train_error = np.amin(train_data_error)
    min_test_error = np.amin(test_data_error)

    print("\nThe regression line is of the form:\n insurance = w0 + w1*age + w2*bmi + w3*children")    
    result = [b_sum/num_samples, w1_sum/num_samples, w2_sum/num_samples, w3_sum/num_samples]
    print("\nWeights of the independent variables are: ")
    print("w0 = ",result[0])
    print("w1 = ",result[1])
    print("w2 = ",result[2])
    print("w3 = ",result[3])
    
    print("\nMean of accuracy prediction of training data: ",train_mean_err)
    print("Variance of accuracy of prediction of training data: ",var_train_error)
    print("Minimum error of training data: ",min_train_error)
    
    print("\nMean of accuracy prediction of testing data: ",test_mean_err)
    print("Variance of accuracy of prediction of testing data: ",var_test_error)
    print("Minimum error of testing data: ",min_test_error)
    
    
