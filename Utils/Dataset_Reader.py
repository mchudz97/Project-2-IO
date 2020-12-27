import pandas as pd
from pandas import DataFrame

EXCEL_TYPES = ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def return_numerical_columns(df: DataFrame, result_column: DataFrame,  exclude: [] = None):
    df_num = df.select_dtypes(include=NUMERICS)
    if exclude:
        for ex in exclude:
            df_num = df_num.drop(ex, axis=1)

    df_num = df_num.join(result_column)
    return df_num


def return_str_columns(df: DataFrame, result_column: DataFrame, include: [] = None):
    return df.drop(return_numerical_columns(df, result_column, exclude=include).columns.values, axis=1)\
        .join(result_column)


class DatasetReader:
    def __init__(self, file: str):
        self.data: DataFrame
        try:
            if file.split('.')[-1] in EXCEL_TYPES:
                self.data = pd.read_excel(file, engine='openpyxl')
            elif file.split('.')[-1] == 'csv':
                self.data = pd.read_csv(file)
            else:
                raise Exception('Cannot handle that type of file.')

        except IndexError:
            raise Exception('File must match [path].[extension]')

    def get_column_names_except_first(self):
        return self.data.columns[1:].values

    def return_df_without_first_column(self):
        return self.data[self.get_column_names_except_first()].copy()





