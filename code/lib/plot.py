import matplotlib.pyplot as plt
import numpy as np


def plot_history(history):
    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error')
    plt.plot(history.epoch, np.array(history.history['loss']),
             label='Train Loss')
    plt.plot(history.epoch, np.array(history.history['val_loss']),
             label='Val Loss')
    plt.plot(history.epoch, np.array(history.history['acc']),
             label='Train Acc')
    plt.plot(history.epoch, np.array(history.history['val_acc']), label='Val Acc')
    plt.legend()
    plt.show()


def plot_roc(fpr, tpr, auc):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('roc curve')
    plt.legend(loc="lower right")
    plt.show()
