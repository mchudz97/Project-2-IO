import time

from sklearn.svm import SVC


class SupportVectorMachines:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'SVM'
        self.clf = SVC()
        start = time.time()
        self.clf.fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.clf.predict(X_test)
