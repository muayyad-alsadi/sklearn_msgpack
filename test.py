from __future__ import print_function

import time
import numpy as np

from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

import sklearn_msgpack

X, y = datasets.make_classification(
    n_samples=1000, n_features=20,
    n_informative=2, n_redundant=10,
    random_state=0)
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.2, random_state=0)

clf = MLPClassifier(hidden_layer_sizes=(40, 20), max_iter=1000)
print("training ...")
t=time.time()
clf.fit(X_train, y_train)
print("done in {} seconds".format(time.time()-t))
print("saving ...")
t=time.time()
sklearn_msgpack.save_to_file('tmp.mpack', clf)
print("done in {} seconds".format(time.time()-t))
print("restoring ...")
clf2 = sklearn_msgpack.load_from_file('tmp.mpack')
print("done in {} seconds".format(time.time()-t))
y1=clf.predict(X_test)
acc1=accuracy_score(y_test, y1)
y2=clf2.predict(X_test)
acc2=accuracy_score(y_test, y1)
print("accuracy of saved model is {} and restored is {}".format(acc1, acc2))
assert(np.array_equal(y1, y2))
