# -*- coding: utf-8 -*-
"""
Created on Fri Mar  8 14:23:23 2019

@author: dishant
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# importing data set
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

# text preprocessing
# remove uncessary words
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
corpus=[]
for i in range(0,31962):
    #tweet=re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
    tweet=re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])', ' ', train['tweet'][i])
    tweet=tweet.lower()
    tweet=tweet.split()
    ps=PorterStemmer()
    #stopwords=stopwords.append(ss)   
    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('English'))]
    tweet=' '.join(tweet)
    corpus.append(tweet)

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer(max_features=17197)
X=cv.fit_transform(corpus).toarray()    

Y=train['label'].values

#train test split
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=25)
classifier.fit(X_train, Y_train)

#predicting
Y_pred=classifier.predict(X_test)

#confusion Matrix
from sklearn.metrics import confusion_matrix
CM=confusion_matrix(Y_test,Y_pred)

from sklearn.metrics import roc_auc_score
print(roc_auc_score(Y_test, Y_pred))

from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, Y_pred))

corpus1=[]
for i in range(0,17197):
    #tweet=re.sub('[^a-zA-Z]', ' ', train['tweet'][i])
    tweet=re.sub('(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])', ' ', test['tweet'][i])
    tweet=tweet.lower()
    tweet=tweet.split() 
    ps=PorterStemmer()
    #stopwords=stopwords.append(ss)   
    tweet=[ps.stem(word) for word in tweet if not word in set(stopwords.words('English'))]
    tweet=' '.join(tweet)
    corpus1.append(tweet)

#Bag of words model
from sklearn.feature_extraction.text import CountVectorizer 
cv1=CountVectorizer(max_features=17197)
X_pred=cv1.fit_transform(corpus1).toarray()

Y_pred_test=classifier.predict(X_pred)


Y_pred_test=np.array(Y_pred_test, dtype=str)
Y_pred_test.to_csv('submission1')
