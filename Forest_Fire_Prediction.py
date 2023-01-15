#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.neural_network import MLPClassifier,MLPRegressor

from sklearn.tree import DecisionTreeRegressor as dtr
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import r2_score

import warnings
warnings.filterwarnings(action='ignore')


# In[3]:


df=pd.read_csv('forestfires.csv')


# In[4]:


df


# In[5]:


df.info()


# In[6]:


df['rain'].value_counts()


# In[7]:


df.count()


# In[8]:


df['month'].unique()


# In[9]:


df['day'].unique()


# In[10]:


df.head(10)


# In[11]:


print(df.isnull().sum())


# In[12]:


missing_values = ["unknown"]

# reading the data again, with the defined non-standard missing value
new_data = pd.read_csv('forestfires.csv', na_values = missing_values)

print(new_data.isnull().sum())


# #### **Preprocessing**
# 
# categorical encoding. Text data included with the numeric data(Month & Days). So we need to encode that in some numeric form before splitting the train test data.

# In[13]:


df=df.drop(['X','Y','month','day'],axis=1)


# In[14]:


def preprocessing(df,task):
  df=df.copy()
  if task=='Regression':
    Y=df['area']
  elif task=='Classification':
    Y=df['area'].apply(lambda x: 1 if x>0 else 0)

  X=df.drop('area',axis=1)

  X_train,X_test,Y_train,Y_test=train_test_split(X,Y,train_size=0.65,shuffle=False,random_state=0)

  scaler=StandardScaler()
  scaler.fit(X_train)

  X_train=pd.DataFrame(scaler.transform(X_train),columns=X.columns)
  X_test=pd.DataFrame(scaler.transform(X_test),columns=X.columns)

  return X_train,X_test,Y_train,Y_test


# In[15]:


X_train,X_test,Y_train,Y_test=preprocessing(df,task='Classification')


# In[16]:


X_train.head()


# In[17]:


import seaborn as sns
plt.figure(figsize=(12,10))
corr=X_train.corr()
sns.heatmap(corr, annot=True, cmap=plt.cm.CMRmap_r )
plt.show()


# In[18]:


def correlation(dataset,threshold):
  col_corr=set()
  corr_matrix=dataset.corr()
  for i in range(len(corr_matrix.columns)):
    for j in range(i):
      if(abs(corr_matrix.iloc[i,j])>threshold):
        colname=corr_matrix.columns[i]
        col_corr.add(colname)
  return col_corr


# In[19]:


corr_features=correlation(X_train,0.7)
len(set(corr_features))


# In[20]:


corr_features


# In[21]:


X_train.drop(corr_features,axis=1).head()


# In[22]:


X_test.drop(corr_features,axis=1).head()


# In[23]:


nn_classifier_model=MLPClassifier(activation='relu',hidden_layer_sizes=(16,16),n_iter_no_change=100,solver='adam')
nn_classifier_model.fit(X_train,Y_train)


# In[24]:


print('MLP Classifier Accuracy, {:.5f}%'.format(nn_classifier_model.score(X_test,Y_test)*100))


# In[25]:


nn_classifier_model.predict_proba(X_test[:10])


# In[ ]:


Applying PCA


# In[28]:


from sklearn.decomposition import PCA

# apply the PCA for feature for feature reduction
pca = PCA(n_components=0.95)
pca.fit(X_train)
PCA_X_train = pca.transform(X_train)
PCA_X_test = pca.transform(X_test)


# In[30]:


from sklearn.neural_network import MLPClassifier 

# define and train an MLPClassifier named mlp on the given data
mlp = MLPClassifier(hidden_layer_sizes=(50,200,50), max_iter=300, activation='relu', solver='adam', random_state=1)
mlp.fit(PCA_X_train, Y_train)


# In[39]:


from sklearn.metrics import accuracy_score, mean_squared_error

# print the training error and MSE
print("Training error: %f" % mlp.loss_curve_[-1])
print("Training set score: %f" % mlp.score(PCA_X_train, Y_train))
print("Test set score: %f" % mlp.score(PCA_X_test, Y_test))



# In[ ]:




