import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import numpy as np
import itertools


#-------------------Confusion Matrix plot--------------------------
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          savename = 'matrix.png',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=55)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()


def print_sesp(cm, class_name):
    t = len(class_name)
    for i in range(t):
        print('\n', class_name[i], ':')
        Se = cm[i][i] / np.sum(cm[i, :])
        Sp = 1 - ((np.sum(cm[:, i]) - cm[i][i]) / (np.sum(cm) - np.sum(cm[i, :])))
        print('Se = ' + str(Se))
        print('Sp = ' + str(Sp))
