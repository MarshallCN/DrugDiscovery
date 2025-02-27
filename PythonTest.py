#coding=utf-8
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors

df = pd.read_csv("FCFP0.csv")
df = df.drop(columns="row ID")

X = df[df.columns[list(df.columns).index('bitvector0'):]] #fingerprint vectors
y = df[df.columns[:list(df.columns).index('bitvector0')]] #all the classes as targets

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

res = []
for i in list(y.columns):
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train[i])
    confidence = clf.score(X_test, y_test[i])
    res.append(confidence)
    print(i,confidence)
print("Average:",np.average(res))