import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from Utils.Dataset_Reader import DatasetReader, return_numerical_columns, return_str_columns


def reduce_null_columns(data: pd.DataFrame, max_percentage_of_nulls: float):
    df = data.copy()
    print(f'Reduced from {df.shape[1]}')
    for column in df:
        if df[column].isna().sum() / df[column].shape[0] >= max_percentage_of_nulls:
            df = df.drop(column, axis=1)

    print(f'to {df.shape[1]} columns')

    return df


def remove_nulls_from_y(data: pd.DataFrame, y_name):
    return data.dropna(subset=[y_name])


def fill_nulls_with_medians(df: pd.DataFrame, numeric_df: pd.DataFrame, y_name: str):
    df_num = numeric_df.copy()
    for col in numeric_df.columns.values:
        if col == y_name:
            continue
        if len(df.groupby(y_name)[col]) >= 2:
            df_num[col] = df[col].fillna(df.groupby(y_name)[col].transform('mean'))
        else:
            df_num[col] = df[col].fillna(df[col].median)

    return df_num


def fill_nulls_with_word(str_df: pd.DataFrame, word: str):
    return str_df.fillna(word)


def create_encoded_df(df_str: pd.DataFrame):

    def ohe_name_parser(ohe_names: [str], df_pre: pd.DataFrame):
        parsed_names = []
        for name in ohe_names.copy():

            class_num = name.split('_', 1)[0]
            addition = name.split('_', 1)[1]
            column_name = df_pre.columns[int(class_num[1:])]
            parsed_names.append(column_name + '_' + addition)

        return parsed_names

    ohe = OneHotEncoder()
    ohe.fit(df_str)
    ohe_df = pd.DataFrame(ohe.transform(df_str).toarray())
    names = ohe.get_feature_names()
    ohe_df.columns = ohe_name_parser(names, df_str)
    return ohe_df


# d = DatasetReader('..\\dataset.xlsx').return_df_without_first_column()
# d = reduce_null_columns(d, .80)
# d = remove_nulls_from_y(d, 'SARS-Cov-2 exam result')
# dn = fill_nulls_with_medians(d, return_numerical_columns(d, d['SARS-Cov-2 exam result']), 'SARS-Cov-2 exam result')
# ds = return_str_columns(d, d['SARS-Cov-2 exam result'])
#
# print(dn.shape)
# for c in ds:
#     print(ds[c].unique())
# ds = fill_nulls_with_word(ds, 'not_tested')
# print(dn.isnull().sum())
# print(create_encoded_df(ds))
