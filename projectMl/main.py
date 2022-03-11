import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR

np.random.seed(0)
m = 100
X = np.linspace(0, 10, m).reshape(m, 1)
y = X + np.random.randn(m, 1)
plt.scatter(X, y)

#model = SVR(C=100)
model = LinearRegression()
model.fit(X, y)
model.score(X, y)
predictions = model.predict(X)

plt.scatter(X, y)
plt.plot(X, predictions, c='r')
plt.show()
