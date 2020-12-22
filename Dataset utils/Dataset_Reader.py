import pandas as pd

EXCEL_TYPES =  ['xls', 'xlsx', 'xlsm', 'xlsb', 'odf', 'ods', 'odt']


class DatasetReader:
    def __init__(self, file: str):
        try:
            if file.split('.')[1] in EXCEL_TYPES:
                self.data = pd.read_excel(file)
            elif file.split('.')[1] == 'csv':
                self.data = pd.read_csv(file)
            else:
                raise Exception('Cannot handle that type of file.')

        except IndexError:
            raise Exception('File must match [name].[extension]')

