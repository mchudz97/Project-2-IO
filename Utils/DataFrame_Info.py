import pandas as pd
from matplotlib import pyplot as plt

def info_about_column(df: pd.DataFrame, column_identity):
    column = df[column_identity]

    return {
        'Max': column.max(),
        'Min': column.min(),
        'Avg': column.mean(),
        'Freq': column.value_counts().to_dict()
    }


def data_type_plot(num_df: pd.DataFrame, str_df: pd.DataFrame):
    labels = ['numeric', 'string']
    amount =[num_df.shape[1], str_df.shape[1]]
    fig, ax = plt.subplots()
    ax.pie(amount, labels=labels, autopct='%.2f')
    ax.axis('equal')
    plt.show()


def null_amount_plot(df: pd.DataFrame):
    data_amount: pd.DataFrame = df.isna().sum() / df.shape[0]
    data_amount.sort_values(ascending=False, inplace=True)
    fig, ax = plt.subplots()
    ax.set_ylabel('None amount')
    ax.bar(list(range(len(data_amount))), data_amount)
    plt.show()
