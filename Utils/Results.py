
def prediction_correctness(y_test, y_pred):
    correctly_predicted = sum(z[0] == z[1] for z in zip(y_test, y_pred))

    return correctly_predicted/len(y_test)
