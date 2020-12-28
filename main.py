from Algorithms.Association_Rules import AssociationRules
from Algorithms.Decision_Tree import DecisionTree
from Algorithms.N_Neighbors import NN
from Algorithms.Naive_Bayes import NaiveBayes
from Algorithms.Neural_Network import NeuralPerceptron, MultiLayerPerceptron
from Algorithms.Random_Forest import RandomForest
from Algorithms.Support_Vector_Machines import SupportVectorMachines
from Utils.DataFrame_Info import null_amount_plot, data_type_plot, column_info_plot, info_about_string_column
from Utils.Data_Preprocessing import reduce_null_columns, remove_nulls_from_y, fill_nulls_with_medians, \
    fill_nulls_with_word, create_encoded_df
from Utils.Data_Separator import DataSeparator
from Utils.Dataset_Reader import DatasetReader, return_numerical_columns, return_str_columns
from Utils.Results import prediction_correctness

not_numeric = ['Patient addmited to regular ward (1=yes, 0=no)',
               'Patient addmited to semi-intensive unit (1=yes, 0=no)',
               'Patient addmited to intensive care unit (1=yes, 0=no)']

interesting_columns = ['Patient addmited to regular ward (1=yes, 0=no)',
                       'Patient addmited to semi-intensive unit (1=yes, 0=no)',
                       'Patient addmited to intensive care unit (1=yes, 0=no)',
                       'Rhinovirus/Enterovirus', 'SARS-Cov-2 exam result']

Y_NAME = 'SARS-Cov-2 exam result'

data = DatasetReader('dataset.xlsx').return_df_without_first_column()
null_amount_plot(data)
data = reduce_null_columns(data, .9)
data = remove_nulls_from_y(data, Y_NAME)
data_numerical = fill_nulls_with_medians(data,
                                         return_numerical_columns(data,
                                                                  data[Y_NAME],
                                                                  not_numeric),
                                         Y_NAME)


# column_info_plot(data_numerical, [Y_NAME])

ds = return_str_columns(data, data[Y_NAME], not_numeric)
data_type_plot(data_numerical, ds)
data_separator_numeric = DataSeparator(data_numerical, .66, Y_NAME)
nb = NaiveBayes(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('nb')
print(prediction_correctness(data_separator_numeric.y_test, nb.predict(data_separator_numeric.X_test)))
dt = DecisionTree(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('dt')
print(prediction_correctness(data_separator_numeric.y_test, dt.predict(data_separator_numeric.X_test)))
n3 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 3)
n5 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 5)
n8 = NN(data_separator_numeric.X_train, data_separator_numeric.y_train, 8)


print(prediction_correctness(data_separator_numeric.y_test, n3.predict(data_separator_numeric.X_test)))
print(prediction_correctness(data_separator_numeric.y_test, n5.predict(data_separator_numeric.X_test)))
print(prediction_correctness(data_separator_numeric.y_test, n8.predict(data_separator_numeric.X_test)))
neural_p = NeuralPerceptron(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('neural_p')
print(prediction_correctness(data_separator_numeric.y_test, neural_p.predict(data_separator_numeric.X_test)))
mlp = MultiLayerPerceptron(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('mlp')
print(prediction_correctness(data_separator_numeric.y_test, mlp.predict(data_separator_numeric.X_test)))
rf = RandomForest(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('rf')
print(prediction_correctness(data_separator_numeric.y_test, rf.predict(data_separator_numeric.X_test)))
svm = SupportVectorMachines(data_separator_numeric.X_train, data_separator_numeric.y_train)
print('svm')
print(prediction_correctness(data_separator_numeric.y_test, svm.predict(data_separator_numeric.X_test)))

ds = fill_nulls_with_word(ds, 'not_tested')
print(info_about_string_column(ds, 'Respiratory Syncytial Virus'))
ds_sliced = ds[interesting_columns]
enc = create_encoded_df(ds_sliced)
ar = AssociationRules(enc)
interesting_rules = ar.rules[ar.rules['antecedents'].apply(lambda x: 'Rhinovirus/Enterovirus_detected' in x)
                             & ar.rules['consequents'].apply(lambda x: 'SARS-Cov-2 exam result_negative' in x)] \
    .sort_values('confidence', ascending=False)

print(interesting_rules.head().to_string())
