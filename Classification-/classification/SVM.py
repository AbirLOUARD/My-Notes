from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import svm

import numpy as np
import arff
import pandas as pd


data = arff.load(open('RFID_Features_windows5.arff', 'r'))['data']

X= [i[:9] for i in data]
y= [i[9] for i in data]

#df = pd.DataFrame(data[0])
#df.head()
print('Classes :', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

svm1= svm.SVC(kernel='linear', C=1.0, random_state=0)
svm1.fit(X_train, y_train)
y_pred = svm1.predict(X_test_std)

print('Misclassified instances: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

svm2= svm.SVC(kernel='rbf', C=1000.0, gamma=10, random_state=0)
svm2.fit(X_train_std, y_train)
y_pred = svm2.predict(X_test_std)

print('Misclassified instances: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))


