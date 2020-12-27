from sklearn.svm import SVC


class SupportVectorMachines:
    def __init__(self, X_train, y_train):
        self.clf = SVC()
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)
