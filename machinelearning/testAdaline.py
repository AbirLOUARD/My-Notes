import Adaline as a
import AdalineSGD as aSGD
import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df = shuffle(df)

y_train = df.iloc[0:150, 4].values
y_train = np.where(y_train == 'Iris-setosa', -1, 1)

X_train = df.iloc[0:150, :4].values

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(16, 4))

ada1 = a.AdalineGD(n_iter=15, eta=0.01).fit(X_train, y_train)
ax[0].plot(range(1, len(ada1.cost_) + 1), np.log10(ada1.cost_), marker='o')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('log(Sum-squared-error)')
ax[0].set_title('Adaline - Learning rate 0.01')

ada2 = a.AdalineGD(n_iter=15, eta=0.001).fit(X_train, y_train)
ax[1].plot(range(1, len(ada2.cost_) + 1), np.log10(ada2.cost_), marker='o')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('log(Sum-squared-error)')
ax[1].set_title('Adaline - Learning rate 0.001')

ada3 = a.AdalineGD(n_iter=15, eta=0.0001).fit(X_train, y_train)
ax[2].plot(range(1, len(ada3.cost_) + 1), ada3.cost_, marker='o')
ax[2].set_xlabel('Epochs')
ax[2].set_ylabel('Sum-squared-error')
ax[2].set_title('Adaline - Learning rate 0.0001')

adaSGD = aSGD.AdalineSGD(n_iter=15, eta=0.0001).fit(X_train, y_train)
ax[3].plot(range(1, len(adaSGD.cost_) + 1), adaSGD.cost_, marker='o')
ax[3].set_xlabel('Epochs')
ax[3].set_ylabel('Sum-squared-error')
ax[3].set_title('AdalineSGD - Learning rate 0.0001')

plt.tight_layout()
# plt.savefig('./adaline_1.png', dpi=300)
plt.show()