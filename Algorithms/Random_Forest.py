import time

from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'RandF'
        self.cls = RandomForestClassifier()
        start = time.time()
        self.cls.fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.cls.predict(X_test)
