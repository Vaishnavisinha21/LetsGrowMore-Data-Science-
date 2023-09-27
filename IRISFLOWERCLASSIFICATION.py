#!/usr/bin/env python
# coding: utf-8

# # Import Modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Loading The Dataset

# In[3]:


df=pd.read_csv("C:/Users/vaish/Downloads/Iris.csv")
df.head()


# # Preprocessing the Dataset

# In[4]:


# Delete the column
df=df.drop(columns="Id")


# In[5]:


df.head()


# In[6]:


# to display stats about the data
df.describe()


# In[7]:


# basic  info of datatype in dataset
df.info()


# In[8]:


# Display the number of samples on each class
df["Species"].value_counts()


# In[9]:


# Check for the null values
df.isnull().sum()


# # Exploratory Data Analysis

# In[10]:


# histograms
df['SepalLengthCm'].hist()


# In[11]:


df['SepalWidthCm'].hist()


# In[12]:


df['PetalLengthCm'].hist()


# In[13]:


df['PetalWidthCm'].hist()


# In[16]:


sns.pairplot(df,hue='Species')


# # Correlation Matrix

# In[17]:


# Compute the correlation matrix
df.corr()


# In[18]:


# display the correlation matrix using a heatmap
corr = df.corr()
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(corr, annot=True, ax=ax, cmap='coolwarm')


# # Model Training

# In[19]:


# Split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X = df.drop(columns=['Species'])
Y = df['Species']
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)


# # Model 1

# In[20]:


# Logistic Regression Model
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)
accuracy_logreg = model1.score(x_test, y_test) * 100
print("Accuracy (Logistic Regression): ", accuracy_logreg)


# # Model 2

# In[21]:


# K-nearest Neighbours Model (KNN)
from sklearn.neighbors import KNeighborsClassifier
model2 = KNeighborsClassifier()
model2.fit(x_train, y_train)
accuracy_knn = model2.score(x_test, y_test) * 100
print("Accuracy (KNN): ", accuracy_knn)


# # Model 3

# In[22]:


# Decision Tree Model
from sklearn.tree import DecisionTreeClassifier
model3 = DecisionTreeClassifier()
model3.fit(x_train, y_train)
accuracy_decision_tree = model3.score(x_test, y_test) * 100
print("Accuracy (Decision Tree): ", accuracy_decision_tree)


# # Project Report

# In[23]:


# Model Comparison - Visualization
models = ['Logistic Regression', 'KNN', 'Decision Tree']
accuracies = [accuracy_logreg, accuracy_knn, accuracy_decision_tree]

plt.bar(models, accuracies, color=['blue', 'green', 'orange'])
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("Model Comparison - Accuracy")
plt.ylim([0, 100])
plt.show()


# In[ ]:




