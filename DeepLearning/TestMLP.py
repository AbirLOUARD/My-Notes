import MLP as mlp
import MNIST as mnist
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import sys

X_train, y_train = mnist.load_mnist('mnist/', kind='train')
X_test, y_test = mnist.load_mnist('mnist/', kind='t10k')

#print("Training")
#mlp_ppn = mlp.NeuralNetMLP(n_output=10, n_features=784, epochs=200, eta=0.00001, l1=0.005, l2=0.001)
#mlp_ppn.fit(X_train, y_train, print_progress=True)

#print('Test')
#y_pred = mlp_ppn.predict(X_test)
#print("Misclassified samples : %d" % (y_test != y_pred).sum())
#print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))

a = np.arange(5)
b = a
print('a & b', np.may_share_memory(a, b))


a = np.arange(5)
print('a & b', np.may_share_memory(a, b))
nn = mlp.NeuralNetMLP(n_output=10,
                  n_features=X_train.shape[1],
                  n_hidden=50,
                  l2=0.1,
                  l1=0.0,
                  epochs=1000,
                  eta=0.001,
                  alpha=0.001,
                  decrease_const=0.00001,
                  minibatches=50,
                  shuffle=True,
                  random_state=1)
nn.fit(X_train, y_train, print_progress=True)
plt.plot(range(len(nn.cost_)), nn.cost_)
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs * 50')
plt.tight_layout()
# plt.savefig('./figures/cost.png', dpi=300)
plt.show()

batches = np.array_split(range(len(nn.cost_)), 1000)
cost_ary = np.array(nn.cost_)
cost_avgs = [np.mean(cost_ary[i]) for i in batches]
plt.plot(range(len(cost_avgs)), cost_avgs, color='red')
plt.ylim([0, 2000])
plt.ylabel('Cost')
plt.xlabel('Epochs')
plt.tight_layout()
#plt.savefig('./figures/cost2.png', dpi=300)
plt.show()

y_train_pred = nn.predict(X_train)

if sys.version_info < (3, 0):
    acc = ((np.sum(y_train == y_train_pred, axis=0)).astype('float') /
           X_train.shape[0])
else:
    acc = np.sum(y_train == y_train_pred, axis=0) / X_train.shape[0]

print('Training accuracy: %.2f%%' % (acc * 100))

y_test_pred = nn.predict(X_test)

if sys.version_info < (3, 0):
    acc = ((np.sum(y_test == y_test_pred, axis=0)).astype('float') /
           X_test.shape[0])
else:
    acc = np.sum(y_test == y_test_pred, axis=0) / X_test.shape[0]

print('Test accuracy: %.2f%%' % (acc * 100))