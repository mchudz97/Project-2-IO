from pandas import DataFrame

from Utils.Dataset_Reader import DatasetReader, return_numerical_columns


def reduce_null_columns(data: DataFrame, max_percentage_of_nulls: float):
    df = data.copy()
    print(f'Reduced from {df.shape[1]}')
    for column in df:

        if df[column].isna().sum() / df[column].shape[0] >= max_percentage_of_nulls:
            df = df.drop(column, axis=1)

    print(f'to {df.shape[1]} columns.')
    return df


def remove_nulls_from_y(data: DataFrame, y_name):
    return data.dropna(subset=[y_name])


def fill_nulls_with_medians(df: DataFrame, numeric_df: DataFrame, y_name):
    df_num = numeric_df.copy()
    for col in numeric_df.columns.values:
        df_num[col] = df[col].fillna(df.groupby(y_name)[col].transform('mean'))

    return df_num


# d = DatasetReader('..\\dataset.xlsx').return_df_without_first_column()
# reduce_null_columns(d, .1)
# d = remove_nulls_from_y(d, 'SARS-Cov-2 exam result')
# return_numerical_columns(d)
# print(fill_nulls_with_medians(d, return_numerical_columns(d), 'SARS-Cov-2 exam result'))
