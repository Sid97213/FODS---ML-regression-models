# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:50:47 2020

@author: Siddhi
"""
import pandas as pd
from random import seed
import numpy as np
from sklearn import linear_model

#Split dataset into training and testing set
def train_test_split(dataframe,split=0.70):
    train_size = int(split * len(dataframe))    #print(train_size)
    test_size = len(dataframe) - train_size     #print(test_size)
    dataframe = dataframe.sample(frac=1)        #shuffle rows of the dataframe     
    train = dataframe[:train_size]              #copy first 70% elements into train
    test = dataframe[-test_size:]               #copy last 30% elements into test
    return train,test

def Regress(df):
    train, test = train_test_split(dataframe)   
    
    X = train.loc[:,'bias':'children']
    Y = train.loc[:,'charges']
    
    regr = linear_model.LinearRegression()
    regr.fit(X, Y)
    #print(regr.score(X,Y))
    return regr.coef_

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
    
    num_samples = 20
    rows,cols = (num_samples,4)
    w0,w1,w2,w3 = (0,0,0,0)
    weight = [[0]*cols]*rows
    for i in range(0,num_samples):
        weight[i] = Regress(dataframe)
        w0 += weight[i][0]
        w1 += weight[i][1]
        w2 += weight[i][2]
        w3 += weight[i][3]
    
    print("\nThe regression line is of the form:\n insurance = w0 + w1*age + w2*bmi + w3*children")    
    result = [w0/num_samples, w1/num_samples, w2/num_samples, w3/num_samples]
    print("\nWeights of the independent variables are: ")
    print("w0 = ",result[0])
    print("w1 = ",result[1])
    print("w2 = ",result[2])
    print("w3 = ",result[3])
    