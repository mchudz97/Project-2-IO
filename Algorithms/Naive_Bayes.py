import time

from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'NB'
        self.nb = GaussianNB()
        start = time.time()
        self.nb.fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.nb.predict(X_test)
