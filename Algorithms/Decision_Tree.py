from sklearn.tree import DecisionTreeClassifier


class DecisionTree:
    def __init__(self, X_train, y_train):
        self.tree_clf = DecisionTreeClassifier().fit(X_train, y_train)

    def predict(self, X_test):
        return self.tree_clf.predict(X_test)