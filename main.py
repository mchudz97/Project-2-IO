from Algorithms.Decision_Tree import DecisionTree
from Algorithms.N_Neighbors import NN
from Algorithms.Naive_Bayes import NaiveBayes
from Algorithms.Neural_Network import NeuralPerceptron, MultiLayerPerceptron
from Utils.Data_Preprocessing import reduce_null_columns, remove_nulls_from_y, fill_nulls_with_medians
from Utils.Data_Separator import DataSeparator
from Utils.Dataset_Reader import DatasetReader, return_numerical_columns
from Utils.Results import prediction_correctness

not_numeric = ['Patient addmited to regular ward (1=yes, 0=no)',
               'Patient addmited to semi-intensive unit (1=yes, 0=no)',
               'Patient addmited to intensive care unit (1=yes, 0=no)']

data = DatasetReader('dataset.xlsx').return_df_without_first_column()
data = reduce_null_columns(data, .95)
data = remove_nulls_from_y(data, 'SARS-Cov-2 exam result')
data_numerical = fill_nulls_with_medians(data,
                                         return_numerical_columns(data,
                                                                  data['SARS-Cov-2 exam result'],
                                                                  not_numeric),
                                         'SARS-Cov-2 exam result')

data_separator_numeric = DataSeparator(data_numerical, .66, 'SARS-Cov-2 exam result')

nb = NaiveBayes(data_separator_numeric.X_train, data_separator_numeric.y_train)
print(prediction_correctness(data_separator_numeric.y_test, nb.predict(data_separator_numeric.X_test)))
dt = DecisionTree(data_separator_numeric.X_train, data_separator_numeric.y_train)
print(prediction_correctness(data_separator_numeric.y_test, dt.predict(data_separator_numeric.X_test)))
n3 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 3)
n5 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 5)
n8 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 8)
print(prediction_correctness(data_separator_numeric.y_test, n3.predict(data_separator_numeric.X_test)))
print(prediction_correctness(data_separator_numeric.y_test, n5.predict(data_separator_numeric.X_test)))
print(prediction_correctness(data_separator_numeric.y_test, n8.predict(data_separator_numeric.X_test)))
neural_p = NeuralPerceptron(data_separator_numeric.X_train, data_separator_numeric.y_train)
print(prediction_correctness(data_separator_numeric.y_test, neural_p.predict(data_separator_numeric.X_test)))
mlp = MultiLayerPerceptron(data_separator_numeric.X_train, data_separator_numeric.y_train)
print(prediction_correctness(data_separator_numeric.y_test, mlp.predict(data_separator_numeric.X_test)))