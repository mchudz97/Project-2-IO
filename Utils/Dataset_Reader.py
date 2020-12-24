import pandas as pd
from pandas import DataFrame

EXCEL_TYPES = ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']
NUMERICS = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']


def return_numerical_columns(df: DataFrame):
    return df.select_dtypes(include=NUMERICS)


def return_str_columns(df: DataFrame):
    return df.drop(return_numerical_columns(df).columns.values, axis=1)


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





