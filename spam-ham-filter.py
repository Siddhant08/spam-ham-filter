# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 11:42:08 2017

@author: shashwats
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pandas as pd
import time

t=time.time()

#read data from csv
readCSV=pd.read_csv('sms-spam-ham.csv')
X=readCSV['Text']
Y=readCSV['Class']

#remove encoding
X_utf=[]
for i in X:
    X_utf.append(unicode(i, errors='ignore'))

#build dictionary to store word count
cv = CountVectorizer()

X=cv.fit_transform(X_utf)

#downscaling frequency using tf-idf rule
tfidf_transformer = TfidfTransformer()

X_train_tfidf=tfidf_transformer.fit_transform(X)

#split dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X_train_tfidf,Y,test_size=0.30,random_state=42)

#training the classifier
print "Multinomial NB Classifier"
mnb = MultinomialNB().fit(X_train,Y_train)

pred=mnb.predict(X_test)

print confusion_matrix(Y_test,pred)
print accuracy_score(Y_test,pred)

#Cross validation
print "Cross validation cv=10"
scores = cross_val_score(mnb, X_train_tfidf, Y, cv=10)
print scores.mean()

print "SVM"
clf= svm.SVC().fit(X_train,Y_train)
pred=mnb.predict(X_test)

print confusion_matrix(Y_test,pred)
print accuracy_score(Y_test,pred)

#Cross validation
print "Cross validation cv=50"
scores = cross_val_score(mnb, X_train_tfidf, Y, cv=10)
print scores.mean()

print "ANN"
ann=MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1).fit(X_train,Y_train)
pred=ann.predict(X_test)

print confusion_matrix(Y_test,pred)
print accuracy_score(Y_test,pred)

#Cross validation
print "Cross validation cv=50"
scores = cross_val_score(mnb, X_train_tfidf, Y, cv=10)
print scores.mean()

print time.time()-t
