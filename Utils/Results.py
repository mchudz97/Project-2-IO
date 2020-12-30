from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt

def prediction_correctness(y_test, y_pred):
    correctly_predicted = sum(z[0] == z[1] for z in zip(y_test, y_pred))

    return correctly_predicted/len(y_test)


def conf_matrixes(alg_results: [tuple], y_test):
    for alg in alg_results:
        print(f'{alg[0]}:\n{confusion_matrix(alg[1], y_test)}')


def compare_results_plot(results: [tuple], y_test):
    prediction_cor = [prediction_correctness(x[1], y_test) for x in results]
    print(prediction_cor)
    # print(y_test)
    labels = [x[0] for x in results]
    fig, ax = plt.subplots()

    ax.bar(list(range(len(prediction_cor))), prediction_cor)
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_xticklabels(labels=labels)
    plt.show()


def time_plot(times: [tuple]):
    labels = [x[0] for x in times]
    times = [x[1] for x in times]
    fig, ax = plt.subplots()
    ax.set_ylabel('time[s]')
    ax.bar(list(range(len(times))), times)
    ax.set_xticks([x for x in range(len(labels))])
    ax.set_xticklabels(labels=labels)
    plt.show()
