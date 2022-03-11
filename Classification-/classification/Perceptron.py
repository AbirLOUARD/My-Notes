
#from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron

import numpy as np
import pandas as pd
import arff
#data= arff.loadarff("RFID_Features_windows5.arff")

data = arff.load(open('RFID_Features_windows5.arff', 'r'))['data']
X= [i[:9] for i in data]
y= [i[9] for i in data]

df = pd.DataFrame(data[0])
df.head()

print('Classes :', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

ppn = Perceptron(max_iter=40, eta0=0.1, random_state=0)
ppn.fit(X_train, y_train)
y_pred = ppn.predict(X_test_std)

print('Misclassified instances: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
