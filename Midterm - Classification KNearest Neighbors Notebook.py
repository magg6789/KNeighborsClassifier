#!/usr/bin/env python
# coding: utf-8

# In[71]:


# Import Python Libraries: NumPy and Pandas
import pandas as pd
import numpy as np
# Import Libraries & modules for data visualization
from pandas.plotting import scatter_matrix
from matplotlib import pyplot


# In[72]:


# Import scikit-Learn module for the algorithm/modeL: Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
# Import scikit-Learn module to split the dataset into train/ test sub-datasets
from sklearn.model_selection import train_test_split


# In[73]:


# Import scikit-Learn module for K-fold cross-validation - algorithm/modeL evaluation & validation
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# Import scikit-Learn module classification report to later use for information about how the system try to classify / lable each record
from sklearn.metrics import classification_report


# # Load the data

# ### Data Set: pima_diabetes.csv

# In[74]:


filename ='C:/Users/miriamgarcia/Downloads/pima_diabetes.csv'
df=pd.read_csv(filename)


# # Preprocess Dataset

# In[75]:


# count the number of NaN values in each column
print (df.isnull().sum())


# In[76]:


#mark zero values as missing or NaN - do not include class
df[[ 'preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age']] = df[['preg' , 'plas' , 'pres' ,'skin', 'test', 'mass', 'pedi', 'age' ]].replace(0,np.NaN)
df=df.fillna(df.mean())
# count the number of NaN values in each column
print (df.isnull().sum())


# In[77]:


# count the number of NaN values in each column
print (df.isnull().sum())


# In[ ]:





# ## Exploratory data analysis (EDA) on the dataset

# In[78]:


# get the dimensions or shape of the dataset
# i.e. number of records / rows X number of variables / columns
print(df.shape)


# In[79]:


#return the first five records / rows of the data set
print(df.head(5))


# In[80]:


#get the data types of all the variables / attributes in the data set
print(df.dtypes)


# In[81]:


#return the summary statistics of the numeric variables / attributes in the data set
print(df.describe())


# In[82]:


#class distribution i.e. how many records are in each class
print(df.groupby('class').size())


# In[83]:


#plot histogram of each numeric variable / attribute in the data set
df.hist(figsize=(12, 8))
pyplot.show()


# In[84]:


# generate density plots of each numeric variable / attribute in the data set
df.plot(kind='density', subplots=True, layout=(3, 3), sharex=False, legend=True, fontsize=1,
figsize=(12, 16))
pyplot.show()


# In[85]:


# generate box plots of each numeric variable / attribute in the data set
df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, figsize=(12,8))
pyplot.show()


# In[86]:


# generate scatter plot matrix of each numeric variable / attribute in the data set
scatter_matrix(df, alpha=0.8, figsize=(15, 15))
pyplot.show()


# ##Separate Dataset into Input & Output NumPy arrays

# In[87]:


# store dataframe values into a numpy array
array = df.values
# separate array into input and output by slicing
# for X(input) [:, 0:8] --> all the rows, columns from 0 - 8 
# these are the independent variables or predictors
X = array[:,0:8]
# for Y(input) [:, 8] --> all the rows, column 8
# this is the value we are trying to predict
Y = array[:,8]


# In[88]:


test_size =0.33
seed=4


# In[89]:


X_train,X_test,Y_train,Y_test =train_test_split(X,Y,test_size=test_size, random_state=seed)


# # Build and Train the Model

# In[90]:


# build the model
model = KNeighborsClassifier()
# train the model using the training sub-dataset
model.fit(X_train, Y_train)
#print the classification report
predicted = model.predict(X_test)
report = classification_report(Y_test, predicted)
print(report)


# # Score the accuracy of the model

# In[91]:


#score the accuracy leve
result = model.score(X_test, Y_test)
#print out the results
print(("Accuracy: %.3f%%") % (result*100.0))


# # Classify/Predict Model 1

# Use the trained model to predict / classify using the following predictors

# In[92]:


model.predict([[6.0, 110, 68, 15,85,18,0.5,38]])


# # Evaluate the model using the 10-fold cross-validation technique

# In[93]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10
# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 4


# In[94]:


kfold = KFold(n_splits, random_state=seed)
scoring = 'accuracy'


# In[95]:


# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# print the evaluationm results
# result: the average of all the results obtained from the K-fold cross validation
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# using the 10-fold cross-validation to evaluate the model / algorithm, the accuracy of this logistic regression
# model is 71%

# # Predict Model  2

# In[96]:


model.predict([[5, 130, 59, 18,70,16,0.3,31]])


# # Evaluate the model using the 10-fold cross-validation technique

# In[97]:


# evaluate the algorythm
# specify the number of time of repeated splitting, in this case 10 folds
n_splits = 10
# fix the random seed
# must use the same seed value so that the same subsets can be obtained
# for each time the process is repeated
seed = 4


# In[98]:


kfold = KFold(n_splits, random_state=seed)
scoring = 'accuracy'


# In[99]:


# train the model and run K-fold cross validation to validate / evaluate the model
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: %.3f (%.3f)" % (results.mean(), results.std()))


# In[ ]:





# In[ ]:




