import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import csv


def info_about_column(df: pd.DataFrame, column_identity: str):
    column = df[column_identity]

    return [
        column.max(),
        column.min(),
        column.mean()
    ]


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


def column_info_plot(num_df: pd.DataFrame, not_num_col: []):
    info = {'column': [], 'max': [], 'min': [], 'avg': []}
    num_df_cp = num_df.copy()
    for n in not_num_col:
        num_df_cp = num_df_cp.drop(n, axis=1)
    for column in num_df_cp:
        single_info = info_about_column(num_df_cp, column)
        info['column'].append(column)
        info['max'].append(single_info[0])
        info['min'].append(single_info[1])
        info['avg'].append(single_info[2])

    to_df = pd.DataFrame.from_dict(info)
    to_df.to_csv('num_info.csv')
    # x = np.arange(len(num_df_cp.columns))
    # width = .5
    # fig, ax = plt.subplots()
    # ax.bar(x=to_df.index, height=to_df['max'], label='Max', color='r', width=width)
    # ax.bar(x=to_df.index, height=to_df['min'], label='Min', color='b', width=width)
    # ax.bar(x=to_df.index, height=to_df['avg'], label='Mean', color='y', width=width)
    # ax.set_ylabel('value')
    # ax.set_xticks(x)
    # ax.set_xticklabels(labels=[''.join([y[0] for y in x.split()]) for x in num_df_cp.columns.tolist()],
    #                    rotation=90)
    # ax.legend()
    #
    # plt.show()


def info_about_string_column(df: pd.DataFrame, column_identity: str):
    return df[column_identity].value_counts()
