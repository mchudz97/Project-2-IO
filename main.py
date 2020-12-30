from Algorithms.Association_Rules import AssociationRules
from Algorithms.Decision_Tree import DecisionTree
from Algorithms.N_Neighbors import NN
from Algorithms.Naive_Bayes import NaiveBayes
from Algorithms.Neural_Network import NeuralPerceptron, MultiLayerPerceptron
from Algorithms.Random_Forest import RandomForest
from Algorithms.Support_Vector_Machines import SupportVectorMachines
from Utils.DataFrame_Info import null_amount_plot, data_type_plot, column_info_plot, info_about_string_column
from Utils.Data_Preprocessing import reduce_null_columns, remove_nulls_from_y, fill_nulls_with_medians, \
    fill_nulls_with_word, create_encoded_df, select_richest_rows
from Utils.Data_Separator import DataSeparator
from Utils.Dataset_Reader import DatasetReader, return_numerical_columns, return_str_columns
from Utils.Results import prediction_correctness, conf_matrixes, compare_results_plot, time_plot
import numpy as np

not_numeric = ['Patient addmited to regular ward (1=yes, 0=no)',
               'Patient addmited to semi-intensive unit (1=yes, 0=no)',
               'Patient addmited to intensive care unit (1=yes, 0=no)']

interesting_columns = ['Patient addmited to regular ward (1=yes, 0=no)',
                       'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                       'Patient addmited to intensive care unit (1=yes, 0=no)',
                       'Rhinovirus/Enterovirus', 'SARS-Cov-2 exam result']

Y_NAME = 'SARS-Cov-2 exam result'

data = DatasetReader('dataset.xlsx').return_df_without_first_column()
# null_amount_plot(data)
data = reduce_null_columns(data, .95)
data = remove_nulls_from_y(data, Y_NAME)
data_numerical = return_numerical_columns(data, data[Y_NAME], not_numeric)
data_numerical = select_richest_rows(data_numerical, .2)
# column_info_plot(data_numerical, [Y_NAME])
# null_amount_plot(data_numerical)
data_numerical = fill_nulls_with_medians(data_numerical, Y_NAME)
print(data_numerical.shape)

# column_info_plot(data_numerical, [Y_NAME])

ds = return_str_columns(data, data[Y_NAME], not_numeric)
print(data_numerical[Y_NAME].value_counts())
# data_type_plot(data_numerical, ds)
data_sep = DataSeparator(data_numerical, .66, Y_NAME)
nb = NaiveBayes(data_sep.X_train, data_sep.y_train, data_sep.X_test)
dt = DecisionTree(data_sep.X_train, data_sep.y_train, data_sep.X_test)
n3 = NN(data_sep.X_train, data_sep.y_train, data_sep.X_test, 3)
n5 = NN(data_sep.X_train, data_sep.y_train, data_sep.X_test, 5)
n11 = NN(data_sep.X_train, data_sep.y_train, data_sep.X_test, 11)
neural_p = NeuralPerceptron(data_sep.X_train, data_sep.y_train, data_sep.X_test)
mlp = MultiLayerPerceptron(data_sep.X_train, data_sep.y_train, data_sep.X_test)
rf = RandomForest(data_sep.X_train, data_sep.y_train, data_sep.X_test)
svm = SupportVectorMachines(data_sep.X_train, data_sep.y_train, data_sep.X_test)

classifiers = [nb, dt, n3, n5, n11, neural_p, mlp, rf, svm]

results = [x.predicted for x in classifiers]
time_plot([x.time for x in classifiers])
conf_matrixes(results, data_sep.y_test)

compare_results_plot(results, data_sep.y_test)

ds = fill_nulls_with_word(ds, 'not_tested')
print(ds.shape)
ds_sliced = ds[interesting_columns]
enc = create_encoded_df(ds_sliced)
ar = AssociationRules(enc)
interesting_rules = ar.rules[ar.rules['antecedents'].apply(lambda x: 'Rhinovirus/Enterovirus_detected' in x)
                             & ar.rules['consequents'].apply(lambda x: 'SARS-Cov-2 exam result_negative' in x)] \
    .sort_values('confidence', ascending=False)

print(interesting_rules.head().to_string())
