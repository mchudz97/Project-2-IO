from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, X_train, y_train):
        self.nb = GaussianNB()
        self.nb.fit(X_train, y_train)

    def predict(self, X_test):
        return self.nb.predict(X_test)
