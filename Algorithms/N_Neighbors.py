from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.pipeline import Pipeline


class NN:
    def __init__(self, X_train, y_train, neighbors_number):

        self.nca = NeighborhoodComponentsAnalysis()
        self.knn = KNeighborsClassifier(n_neighbors=neighbors_number)
        self.nca_pipe = Pipeline([('nca', self.nca), ('knn', self.knn)])
        self.clf = self.nca_pipe.fit(X_train, y_train.values.ravel())

    def predict(self, X_test):
        return self.clf.predict(X_test)