#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import urllib

import sklearn
import tensorflow as tf
from IPython import get_ipython
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# requirements : 
#     recommended platform : ubuntu
#     python == 3.7
#     pip install pandas
#     pip install numpy
#     pip install sklearn
#     pip install seaborn
#     pip install matplotlib
#get_ipython().run_line_magic('matplotlib', 'inline')


# # DATASET
# Data Source
#     
#     https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation
#     
#     Source:
#     Rajen Bhatt, Abhinav Dhall, rajen.bhatt '@' gmail.com, IIT Delhi.
#     
#     Data Set Information:
# 
#         The skin dataset is collected by randomly sampling B,G,R values from face images of various age groups (young, middle, and old), race groups (white, black, and asian), and genders obtained from FERET database and PAL database. Total learning sample size is 245057; out of which 50859 is the skin samples and 194198 is non-skin samples.
#         
#     Attribute Information:
# 
#         This dataset is of the dimension 245057 * 4 where first three columns are B,G,R (x1,x2, and x3 features) values and fourth column is of the class labels (decision variable y).
#         
#     Relevant Papers:
# 
# 
# 1. Rajen B. Bhatt, Gaurav Sharma, Abhinav Dhall, Santanu Chaudhury, â€œEfficient skin region segmentation using low complexity fuzzy decision tree modelâ€, IEEE-INDICON 2009, Dec 16-18, Ahmedabad, India, pp. 1-4.
#         2. Abhinav Dhall, Gaurav Sharma, Rajen Bhatt, Ghulam Mohiuddin Khan, â€œAdaptive Digital Makeupâ€, in Proc. of International Symposium on Visual Computing (ISVC) 2009, Nov. 30 â€“ Dec. 02, Las Vegas, Nevada, USA, Lecture Notes in Computer Science, Vol. 5876, pp. 728-736.
# 
# 
# 
#         Citation Request:
# 
#         Rajen Bhatt, Abhinav Dhall, 'Skin Segmentation Dataset', UCI Machine Learning Repository
# 
# 
#     

# ### DOWNLOAD DATASET
#     Direct Link : https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt

# #### DOWNLOAD USING PYTHON

# In[2]:


if not os.path.exists('./Skin_NonSkin.txt'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    urllib.request.urlretrieve(url,'./Skin_NonSkin.txt')


# #### Read using pandas

# In[3]:


df = pd.read_csv('Skin_NonSkin.txt', sep='\t',names =['B','G','R','skin'])
df.head()


# #### Check Missing values

# In[4]:


# NO MISSING VALUES
df.isna().sum()


# ### Standardize dataset

# In[5]:


feature = df[df.columns[~df.columns.isin(['skin'])]] #Except Label
label = (df[['skin']] == 1)*1 #Converting to 0 and 1 (this col has values 1 and 2)
feature = feature / 255. #Pixel values range from 0-255 converting between 0-1


# In[6]:


feature.head()


# In[7]:


label.head()


# ### Explore your data
#     Please try to understand the nature of data

# In[8]:


alldf = pd.concat([feature,label], sort=True, axis=1)
alldf


# In[9]:


sample = alldf.sample(1000)


# In[10]:


onlybgr = sample[sample.columns[~sample.columns.isin(['skin'])]]


# In[11]:


sns.pairplot(onlybgr)


# In[12]:


sample_ = sample.copy()
sample_['skin'] = sample.skin.apply(lambda x:{1:'skin',0:'not skin'}.get(x))
sns.pairplot(sample_, hue="skin")
# sample


# In[13]:


sns.pairplot(onlybgr, kind="reg")


# In[14]:


# Lets see how many 0s and 1s
(label == 0).skin.sum(),(label == 1).skin.sum()


# ## SPLIT DATA INTO 5 CROSS - VALIDATION

# In[15]:


x = feature.values
y = label.values


# In[16]:


# We will keep fix test and take 5 cross validation set
# so we will have five different data set
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)


# In[17]:


#Lets see the size of xtrain, xtest
len(xtrain),len(xtest)


# In[18]:


# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks

data = np.concatenate([xtrain,ytrain],axis = 1)
# Observe the shape of array
xtrain.shape,ytrain.shape,data.shape


# In[19]:


# Divide our data to 5 chunks
chunks = np.split(data,5)


# In[20]:


datadict = {'fold1':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold2':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold3':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}}, 
            'fold4':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold5':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},}


# In[21]:


for i in range(5):
    datadict['fold'+str(i+1)]['val']['x'] = chunks[i][:,0:3]
    datadict['fold'+str(i+1)]['val']['y'] = chunks[i][:,3:4]
    
    idx = list(set(range(5))-set([i]))
    X = np.concatenate(itemgetter(*idx)(chunks),0)
    datadict['fold'+str(i+1)]['train']['x'] = X[:,0:3]
    datadict['fold'+str(i+1)]['train']['y'] = X[:,3:4]
    


# In[22]:


def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def readpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


# In[23]:


writepickle(datadict,'data.pkl')


# ## Now You Can Read This Pickle File And Use In Your Project

# In[24]:


data = readpickle('data.pkl')


# ## How to access data
#     Example : Access fold1 data

# In[25]:

# fold 1
fold1 = data['fold1']
fold1_train = fold1['train']
fold1_val = fold1['val']
fold1_test = fold1['test']

xtrain, ytrain = fold1_train['x'],fold1_train['y']
xval, yval = fold1_val['x'], fold1_val['y']
xtest, ytest = fold1_test['x'],fold1_test['y']

xtrain.shape, ytrain.shape

xval.shape, yval.shape

xtest.shape,ytest.shape

# fit the training data
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(xtrain, ytrain.ravel())


# do test
logisticRegr.predict(xtest[0].reshape(1,-1))
logisticRegr.predict(xtest[0:])
predictions = logisticRegr.predict(xtest)
score_test_fold1 = logisticRegr.score(xtest, ytest)
print("TEST 1: ", score_test_fold1)

# do val
logisticRegr.predict(xval[0].reshape(1,-1))
logisticRegr.predict(xval[0:])
predictions = logisticRegr.predict(xval)
score_val_fold1 = logisticRegr.score(xval, yval)
print("VAL 1: ", score_val_fold1)



# fold 2
fold2 = data['fold2']
fold2_train = fold1['train']
fold2_val = fold1['val']
fold2_test = fold1['test']

xtrain, ytrain = fold2_train['x'],fold2_train['y']
xval, yval = fold2_val['x'], fold2_val['y']
xtest, ytest = fold2_test['x'],fold2_test['y']

xtrain.shape, ytrain.shape

xval.shape, yval.shape

xtest.shape,ytest.shape


# fit the training data
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(xtrain, ytrain.ravel())


# do test
logisticRegr.predict(xtest[0].reshape(1,-1))
logisticRegr.predict(xtest[0:])
predictions = logisticRegr.predict(xtest)
score_test_fold2 = logisticRegr.score(xtest, ytest)
print("TEST 2: ", score_test_fold2)

# do val
logisticRegr.predict(xval[0].reshape(1,-1))
logisticRegr.predict(xval[0:])
predictions = logisticRegr.predict(xval)
score_val_fold2 = logisticRegr.score(xval, yval)
print("VAL 2: ", score_val_fold2)



# fold 3
fold3 = data['fold3']
fold3_train = fold3['train']
fold3_val = fold3['val']
fold3_test = fold3['test']

xtrain, ytrain = fold3_train['x'],fold3_train['y']
xval, yval = fold3_val['x'], fold3_val['y']
xtest, ytest = fold3_test['x'],fold3_test['y']

xtrain.shape, ytrain.shape

xval.shape, yval.shape

xtest.shape,ytest.shape


# fit the training data
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(xtrain, ytrain.ravel())


# do test
logisticRegr.predict(xtest[0].reshape(1,-1))
logisticRegr.predict(xtest[0:])
predictions = logisticRegr.predict(xtest)
score_test_fold3 = logisticRegr.score(xtest, ytest)
print("TEST 3: ", score_test_fold3)

# do val
logisticRegr.predict(xval[0].reshape(1,-1))
logisticRegr.predict(xval[0:])
predictions = logisticRegr.predict(xval)
score_val_fold3 = logisticRegr.score(xval, yval)
print("VAL 3: ", score_val_fold3)


# fold 4
fold4 = data['fold4']
fold4_train = fold1['train']
fold4_val = fold1['val']
fold4_test = fold1['test']

xtrain, ytrain = fold4_train['x'],fold4_train['y']
xval, yval = fold4_val['x'], fold4_val['y']
xtest, ytest = fold4_test['x'],fold4_test['y']

xtrain.shape, ytrain.shape

xval.shape, yval.shape

xtest.shape,ytest.shape


# fit the training data
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(xtrain, ytrain.ravel())


# do test
logisticRegr.predict(xtest[0].reshape(1,-1))
logisticRegr.predict(xtest[0:])
predictions = logisticRegr.predict(xtest)
score_test_fold4 = logisticRegr.score(xtest, ytest)
print("TEST 4: ", score_test_fold4)

# do val
logisticRegr.predict(xval[0].reshape(1,-1))
logisticRegr.predict(xval[0:])
predictions = logisticRegr.predict(xval)
score_val_fold4 = logisticRegr.score(xval, yval)
print("VAL 4: ", score_val_fold4)


# fold 5
fold5 = data['fold5']
fold5_train = fold1['train']
fold5_val = fold1['val']
fold5_test = fold1['test']

xtrain, ytrain = fold5_train['x'],fold5_train['y']
xval, yval = fold5_val['x'], fold5_val['y']
xtest, ytest = fold5_test['x'],fold5_test['y']

xtrain.shape, ytrain.shape

xval.shape, yval.shape

xtest.shape,ytest.shape

# fit the training data
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(xtrain, ytrain.ravel())


# do test
logisticRegr.predict(xtest[0].reshape(1,-1))
logisticRegr.predict(xtest[0:])
predictions = logisticRegr.predict(xtest)
score_test_fold5 = logisticRegr.score(xtest, ytest)
print("TEST 5: ", score_test_fold5)

# do val
logisticRegr.predict(xval[0].reshape(1,-1))
logisticRegr.predict(xval[0:])
predictions = logisticRegr.predict(xval)
score_val_fold5 = logisticRegr.score(xval, yval)
print("VAL 5: ", score_val_fold5)

avg_test = (score_test_fold1 + score_test_fold2 + score_test_fold3 + score_test_fold4 + score_test_fold5)/5
print("AVERAGE TEST: ", avg_test)

avg_val = (score_val_fold1 + score_val_fold2 + score_val_fold3 + score_val_fold4 + score_val_fold5)/5
print("AVERAGE VAL: ", avg_val)

# learning_rate = 0.005
# epochs = 50
# batch = 100
# batches = int(xtrain.shape[0] / batch)


# <h1 style="color:green">!!!!!!!!!! NOTES !!!!!!!!!!</h1>
# 
#     This walkthrough is just to make your life easier
#     If you want to use your own ways of doing data processing it is fine
#     You can continue your assignment right-away from below or setup 
#     a project like assignment 1. You can use any library you want

# <h1 style="color:red">!!!!!!!!!! WARNING !!!!!!!!!!</h1>
# 
#     1. DO NOT SUBMIT .DOCX FILE
#     2. NAME YOUR FILE IN FOLLOWING MANNER net_id_homework_2.pdf
#     3. SUBMIT BOTH ASSIGNMENT(OPENED) AND YOUR DROPBOX
#     4. YOUR REPORT SHOULD CONTAIN LINK TO YOUR GITHUB 
#     5. ADD COMMENTS TO INCREASE READIBILITY IN YOUR CODE

# ### Now use above dataset to complete following work

# # Assignment 2
#     You can use any libraires you want, but choose python as your platform
#     
#     1. Implement Logistic Regression on this 5 fold data
#     2. Report Test Accuracy, Val Accuracy on each fold
#        Follow following format
#        ________________________
#        
#             |  ACCURACY   
#        FOLD | VAL | TEST  
#        ________________________
#             |     |
#        1    |  ?? |  ??
#        2    |  ?? |  ??
#        3    |  ?? |  ??
#        4    |  ?? |  ??
#        5    |  ?? |  ??
#        ________________________
#        AVG  |  ?? |  ??
#        
#     3. Report Visualization
#     
#     NOTE :  You must submit two things
#             First : A pdf report with following explanation
#                     - What tools you used and why?
#                     - Metrics as explained in (2)
#                     - Visualization/Graph
#                     - Conclude your experiment
#                     - Add a github repo as report
# 
#             Second : A github repo
#     
