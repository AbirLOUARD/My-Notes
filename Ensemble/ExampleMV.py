import MajorityVote as mv

from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

import numpy as np
import arff
import pandas as pd


data = arff.load(open('RFID_Features_windows5.arff', 'r'))['data']
df = pd.DataFrame(data[0])
df.head()
X= [i[:9] for i in data]
y= [i[9] for i in data]
print('Classes :', np.unique(y))

yBytes = df.iloc[:1000 , -1].values
encoding = 'utf-8'
j = 0
for i in yBytes:
    y.append(str(yBytes[j], encoding))
    j +=1

X_train, X_test , y_train , y_test = train_test_split(X , y , test_size = 0.5 , random_state = 0)


sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

clf1 = clf = RandomForestClassifier(n_estimators=100)

clf2 = AdaBoostClassifier(n_estimators=60,
                                   learning_rate=0.05,
                                   random_state=1)

clf3 = HistGradientBoostingClassifier(max_bins=255, max_iter=100)

pipe1 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf1]])
pipe2 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf2]])
pipe3 = Pipeline([['sc', StandardScaler()],
                  ['clf', clf3]])

clf_labels = ['RandomForestClassifier', 'Adaboost', 'HistGradientBoostingClassifier']

print('10-fold cross validation:\n')
for clf, label in zip([pipe1, clf2, pipe3], clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='f1_macro')

mv_clf = mv.MajorityVoteClassifier(classifiers=[pipe1, clf2, pipe3])

clf_labels += ['Majority Voting']
all_clf = [pipe1, clf2, pipe3, mv_clf]

for clf, label in zip(all_clf, clf_labels):
    scores = cross_val_score(estimator=clf,
                             X=X_train,
                             y=y_train,
                             cv=10,
                             scoring='f1_macro')
    print("F1-Score: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))