#!/usr/bin/env python
# coding: utf-8

# # IRIS DATASET ANALYSIS (CLASSIFICATION)

# # Import Modules

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


## LOADING DATASET


# In[3]:


df = pd.read_csv('Iris.csv')
df.head()


# In[4]:


# Deleting ID column
df = df.drop(columns = ['Id'])
df.head()


# In[5]:


# To display stats about the data
df.describe()


# In[6]:


# To display basic info about datatrype
df.info()


# In[7]:


# To display no of samples on each class
df['Species'].value_counts()


# # Pre processing the dataset

# In[8]:


df.isnull().sum()


# In[9]:


# Exploratory data analysis


# In[10]:


df['SepalLengthCm'].hist()


# In[11]:


df['SepalWidthCm'].hist()


# In[12]:


df['PetalLengthCm'].hist()


# In[13]:


df['PetalWidthCm'].hist()


# In[14]:


df['Species'].hist()


# In[15]:


# scatterplot


# In[16]:


colors = ['red','green','blue']
species = ['Iris-setosa' , 'Iris-versicolor', 'Iris-virginica']


# In[17]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'] , x['SepalWidthCm'] , c=colors[i] , label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.legend()


# In[18]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['PetalLengthCm'] , x['PetalWidthCm'] , c=colors[i] , label=species[i])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.legend()


# In[19]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalLengthCm'] , x['PetalLengthCm'] , c=colors[i] , label=species[i])
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')
plt.legend()


# In[20]:


for i in range(3):
    x = df[df['Species'] == species[i]]
    plt.scatter(x['SepalWidthCm'] , x['PetalWidthCm'] , c=colors[i] , label=species[i])
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')
plt.legend()


# # coorelation

# In[21]:


df.corr()


# In[22]:


corr = df.corr()
fid ,  ax = plt.subplots(figsize=(10,10))
sn.heatmap(corr,annot=True,ax=ax,cmap = 'coolwarm')


# # label encoder

# In[23]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[24]:


df['Species'] = le.fit_transform(df['Species'])
df.head()


# # model Training

# In[25]:


from sklearn.model_selection import train_test_split
# train = 70
# test  = 30
x = df.drop(columns = ['Species'])
y = df['Species']
x_train , x_test ,y_train , y_test = train_test_split(x, y,test_size = 0.30)


# In[26]:


# logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[27]:


model.fit(x_train , y_train)


# # print metric to get performance

# In[28]:


print("accuracy: ",model.score(x_test,y_test)*100)


# # knn - k-nearest neighbours

# In[29]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()


# In[30]:


model.fit(x_train , y_train)


# In[31]:


print("accuracy: ",model.score(x_test,y_test)*100)


# In[32]:


# decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(x_train , y_train)


# In[33]:


print("accuracy: ",model.score(x_test,y_test)*100)


# In[ ]:




