from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import arff
import pandas as pd


data = arff.load(open('RFID_Features_windows5.arff', 'r'))['data']
df = pd.DataFrame(data[0])
df.head()
X= [i[:9] for i in data]
y= [i[9] for i in data]

print('Classes :', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

tree = DecisionTreeClassifier(criterion='entropy', max_depth=3)
#tree = DecisionTreeClassifier(criterion='entropy',ccp_alpha=0.2)
#tree = DecisionTreeClassifier(criterion='gini')
tree.fit(X_train, y_train)
y_pred = tree.predict(X_test_std)

print('Misclassified instances: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
