from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier


class NeuralPerceptron:
    def __init__(self, X_train, y_train):
        self.p = Perceptron()
        self.p.fit(X_train, y_train)

    def predict(self, X_test):
        return self.p.predict(X_test)


class MultiLayerPerceptron:
    def __init__(self, X_train, y_train):
        self.mlp = MLPClassifier()
        self.mlp.fit(X_train, y_train)

    def predict(self, X_test):
        return self.mlp.predict(X_test)
