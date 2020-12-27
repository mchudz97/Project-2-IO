from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, X_train, y_train):
        self.cls = RandomForestClassifier()
        self.cls.fit(X_train, y_train)

    def predict(self, X_test):
        return self.cls.predict(X_test)