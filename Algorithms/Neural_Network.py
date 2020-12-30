import time

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


class NeuralPerceptron:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'Perc'
        self.p = Perceptron()
        start = time.time()
        self.p.fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.p.predict(X_test)


class MultiLayerPerceptron:
    def __init__(self, X_train, y_train, X_test):
        self.name = 'MLP'
        self.mlp = MLPClassifier(max_iter=500)
        start = time.time()
        self.mlp.fit(X_train, y_train)
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.mlp.predict(X_test)
