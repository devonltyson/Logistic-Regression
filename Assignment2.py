### ASSIGNMENT 2 - LOGISTIC REGRESSION
### Devon Tyson

import pandas as pd
import urllib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import os
import numpy as np
from operator import itemgetter
import pickle
import seaborn as sns

# REQUIREMENTS
#   recommended platform: ubuntu
#   python == 3.7
#   pip install pandas
#   pip install numpy
#   pip install sklearn
#   pip install seaborn
#   pip install matplotlib

# DATASET SOURCE
#   https://archive.ics.uci.edu/ml/datasets/Skin+Segmentation

# DOWNLOAD DATASET
#   https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt

if not os.path.exists('./Skin_NonSkin.txt'):
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00229/Skin_NonSkin.txt'
    urllib.request.urlretrieve(url,'./Skin_NonSkin.txt')

# Read using pandas
df = pd.read_csv('Skin_NonSkin.txt', sep='\t',names =['B','G','R','skin'])
df.head()

# Check Missing values
# NO MISSING VALUES
df.isna().sum()


# Standardize dataset
feature = df[df.columns[~df.columns.isin(['skin'])]] #Except Label
label = (df[['skin']] == 1)*1 #Converting to 0 and 1 (this col has values 1 and 2)
feature = feature / 255. #Pixel values range from 0-255 converting between 0-1

feature.head()
label.head()

alldf = pd.concat([feature,label], sort=True, axis=1)

sample = alldf.sample(1000)

onlybgr = sample[sample.columns[~sample.columns.isin(['skin'])]]

sns.pairplot(onlybgr)

sample_ = sample.copy()
sample_['skin'] = sample.skin.apply(lambda x:{1:'skin',0:'not skin'}.get(x))
sns.pairplot(sample_, hue="skin")

sns.pairplot(onlybgr, kind="reg")

# Lets see how many 0s and 1s
(label == 0).skin.sum(),(label == 1).skin.sum()

# SPLIT DATA INTO 5 CROSS - VALIDATION
x = feature.values
y = label.values
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.20, random_state=1)

# 5 Fold Split
# First merge xtrain and ytrain so that we can easily divide into 5 chunks
data = np.concatenate([xtrain,ytrain],axis = 1)

# Divide our data to 5 chunks
chunks = np.split(data,5)

datadict = {'fold1':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold2':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold3':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}}, 
            'fold4':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},
            'fold5':{'train':{'x':None,'y':None},'val':{'x':None,'y':None},'test':{'x':xtest,'y':ytest}},}

for i in range(5):
    datadict['fold'+str(i+1)]['val']['x'] = chunks[i][:,0:3]
    datadict['fold'+str(i+1)]['val']['y'] = chunks[i][:,3:4]
    
    idx = list(set(range(5))-set([i]))
    X = np.concatenate(itemgetter(*idx)(chunks),0)
    datadict['fold'+str(i+1)]['train']['x'] = X[:,0:3]
    datadict['fold'+str(i+1)]['train']['y'] = X[:,3:4]

def writepickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def readpickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

writepickle(datadict,'data.pkl')
data = readpickle('data.pkl')



# start logistic regression segment
lr = LogisticRegression(solver = 'lbfgs')

# fold 1
fold1 = data['fold1']
fold1_train = fold1['train']
fold1_val = fold1['val']
fold1_test = fold1['test']

xtrain_1, ytrain_1 = fold1_train['x'],fold1_train['y']
xval_1, yval_1 = fold1_val['x'], fold1_val['y']
xtest_1, ytest_1 = fold1_test['x'],fold1_test['y']

# fit the training data
lr.fit(xtrain_1, ytrain_1.ravel())

# get accuracy
score_test_fold1 = lr.score(xtest_1, ytest_1)
score_val_fold1 = lr.score(xval_1, yval_1)


# fold 2
fold2 = data['fold2']
fold2_train = fold2['train']
fold2_val = fold2['val']
fold2_test = fold2['test']

xtrain_2, ytrain_2 = fold2_train['x'],fold2_train['y']
xval_2, yval_2 = fold2_val['x'], fold2_val['y']
xtest_2, ytest_2= fold2_test['x'],fold2_test['y']

# fit the training data
lr.fit(xtrain_2, ytrain_2.ravel())

# get accuracy
score_test_fold2 = lr.score(xtest_2, ytest_2)
score_val_fold2 = lr.score(xval_2, yval_2)


# fold 3
fold3 = data['fold3']
fold3_train = fold3['train']
fold3_val = fold3['val']
fold3_test = fold3['test']

xtrain_3, ytrain_3 = fold3_train['x'],fold3_train['y']
xval_3, yval_3 = fold3_val['x'], fold3_val['y']
xtest_3, ytest_3 = fold3_test['x'],fold3_test['y']

# fit the training data
lr.fit(xtrain_3, ytrain_3.ravel())

# get accuracy
score_test_fold3 = lr.score(xtest_3, ytest_3)
score_val_fold3 = lr.score(xval_3, yval_3)


# fold 4
fold4 = data['fold4']
fold4_train = fold4['train']
fold4_val = fold4['val']
fold4_test = fold4['test']

xtrain_4, ytrain_4 = fold4_train['x'],fold4_train['y']
xval_4, yval_4 = fold4_val['x'], fold4_val['y']
xtest_4, ytest_4 = fold4_test['x'],fold4_test['y']

# fit the training data
lr.fit(xtrain_4, ytrain_4.ravel())

# get accuracy
score_test_fold4 = lr.score(xtest_4, ytest_4)
score_val_fold4 = lr.score(xval_4, yval_4)


# fold 5
fold5 = data['fold5']
fold5_train = fold5['train']
fold5_val = fold5['val']
fold5_test = fold5['test']

xtrain_5, ytrain_5 = fold5_train['x'],fold5_train['y']
xval_5, yval_5 = fold5_val['x'], fold5_val['y']
xtest_5, ytest_5 = fold5_test['x'],fold5_test['y']

# fit the training data
lr.fit(xtrain_5, ytrain_5.ravel())

# get accuracy
score_test_fold5 = lr.score(xtest_5, ytest_5)
score_val_fold5 = lr.score(xval_5, yval_5)


# get averages
avg_test = (score_test_fold1 + score_test_fold2 + score_test_fold3 + score_test_fold4 + score_test_fold5)/5
avg_val = (score_val_fold1 + score_val_fold2 + score_val_fold3 + score_val_fold4 + score_val_fold5)/5


# set up output
print("------ ACCURACY -------")
val_test_data = {"VAL" :["--------", score_val_fold1, score_val_fold2, score_val_fold3, score_val_fold4, score_val_fold5, "--------", avg_val],
                 "TEST" :["--------", score_test_fold1, score_test_fold2, score_test_fold3, score_test_fold4, score_test_fold5, "--------",  avg_test]}
val_test_labels = ["---", "1", "2", "3", "4", "5", "---", "AVG"]

print(pd.DataFrame(val_test_data, val_test_labels))