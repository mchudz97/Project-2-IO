import pandas as pd


def info_about_column(df: pd.DataFrame, column_identity):
    column = df[column_identity]

    return {
        'Max': column.max(),
        'Min': column.min(),
        'Avg': column.mean(),
        'Freq': column.value_counts().to_dict()
    }

