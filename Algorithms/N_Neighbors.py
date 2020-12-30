import time

from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline


class NN:
    def __init__(self, X_train, y_train, X_test, neighbors_number):
        self.n = neighbors_number
        self.name = f'{self.n}-n'
        start = time.time()
        self.nca = NeighborhoodComponentsAnalysis()
        self.knn = KNeighborsClassifier(n_neighbors=self.n)
        self.nca_pipe = Pipeline([('nca', self.nca), ('knn', self.knn)])
        self.clf = self.nca_pipe.fit(X_train, y_train.values.ravel())
        self.predicted = self.predict(X_test)
        self.time = self.name, time.time() - start

    def predict(self, X_test):
        return self.name, self.clf.predict(X_test)
