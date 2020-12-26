import pandas as pd
from sklearn.model_selection import train_test_split


class DataSeparator:
    def __init__(self, data: pd.DataFrame, test_size: float, y_name: str):
        x_names = list(data.columns.values)
        x_names.remove(y_name)
        self.to_train, self.to_test = train_test_split(data, test_size=test_size, stratify=data[y_name])
        self.X_train = self.to_train[x_names]
        self.y_train = self.to_train[y_name]
        self.X_test = self.to_test[x_names]
        self.y_test = self.to_test[y_name]
