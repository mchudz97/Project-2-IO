import time

from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'DecTree'
        start = time.time()
        self.tree_clf = DecisionTreeClassifier().fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.tree_clf.predict(X_test)
