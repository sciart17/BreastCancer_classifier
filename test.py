import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
matplotlib.rcParams['font.size'] = 18
matplotlib.rcParams['figure.titlesize'] = 18
matplotlib.rcParams['figure.figsize'] = [9, 7]
matplotlib.rcParams['axes.unicode_minus'] = False

import os
import argparse
import numpy as np
import pycm

import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, roc_auc_score, auc
from tensorflow.keras.models import load_model

import test_utils as utils
from gen_csv import load_train, load_test, load_val
from  dual_path_network import DPN92, DPN98, DPN107, DPN137

tf.random.set_seed(1234)
np.random.seed(1234)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"


batchsz = 32

def preprocess(images,labels):
    imagesids = len(labels)
    X = []
    for id in range(imagesids):
        x = tf.io.read_file(images[id])
        x = tf.image.decode_png(x, channels=1)
        x = tf.image.resize(x, [224, 224])
        x = tf.image.grayscale_to_rgb(x)
        x = tf.cast(x, dtype=tf.float32) / 255.
        X.append(x)

    # X = np.array(X)
    X = tf.convert_to_tensor(X)

    y = tf.cast(labels, dtype=tf.int32)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=3)

    return X, y

def db_preprocess(x,y):
    x = tf.io.read_file(x)
    x = tf.image.decode_png(x, channels=1)  # RGBA
    x = tf.image.resize(x, [224, 224])
    x = tf.image.grayscale_to_rgb(x)

    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.convert_to_tensor(y)
    y = tf.one_hot(y, depth=3)
    return x, y


def run(input_model, save_name):
    # 创建测试集Datset对象
    images3, labels3, table3 = load_test('test', mode='test')
    print('testimages:', len(images3))
    print('testlabels:', len(labels3))
    x_test, y_test = preprocess(images3, labels3)

    db_test = tf.data.Dataset.from_tensor_slices((images3, labels3))
    db_test = db_test.shuffle(500).map(db_preprocess).batch(batchsz)

    print('process data Successfully')

    # -----------predict-----------------------
    model_name = input_model
    model = load_model(model_name)

    test_loss, test_acc = model.evaluate(db_test)
    print('test_loss:', test_loss, 'test_acc:', test_acc, '\n')

    y_test_pred = model.predict(x_test)
    # print('np.argmax(y_test,axis=1)', np.argmax(y_test, axis=1), '\n')
    # print('np.argmax(y_test_pred,axis=1)', np.argmax(y_test_pred, axis=1), '\n')

    test_accuracy = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1))
    print('acc:', test_accuracy, '\n')

    cmcm = pycm.ConfusionMatrix(actual_vector=np.argmax(y_test, axis=1), predict_vector=np.argmax(y_test_pred, axis=1))
    print(cmcm)

    cm_plot_label = ['Benign', 'Malignant', 'Normal']
    cm = confusion_matrix(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1))
    utils.plot_confusion_matrix(cm, cm_plot_label, title='Confusion Matrix for Breast Cancer',
                                savename='matrix_' + save_name + '.png')
    utils.print_sesp(cm, cm_plot_label)
    print('\n---------------------------------------------------------------------')

    # --------------classification report-----------------
    class_outcome = classification_report(np.argmax(y_test, axis=1), np.argmax(y_test_pred, axis=1),
                                          target_names=cm_plot_label)
    print('classification report:', '\n', class_outcome)
    print('\n---------------------------------------------------------------------')

    # -------------------ROC and AUC-----------------------------------
    roc_log = roc_auc_score(y_test, y_test_pred, average='micro')
    print('functionally computed AUC:', roc_log)

    # false-postive-rate fpr = (1-tnr) = 1-specificity
    # true-postive-rate tpr = sensitivity

    fpr, tpr, thresholds = roc_curve(np.ravel(y_test), np.ravel(y_test_pred))
    area_under_curve = auc(fpr, tpr)
    print('manually computed AUC:', area_under_curve)

    # auc curve
    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(area_under_curve))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.savefig('ROC' + save_name + '.svg')
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test model for DDSM images')
    parser.add_argument("--input_model", type=str)
    parser.add_argument("--save_name", type=str)
    args = parser.parse_args()
    run_opts = dict(
    )
    run(args.input_model, args.save_name, **run_opts)
