import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn import metrics
from code.lib.plot import plot_history, plot_roc
from code.lib.result_analyze import result_analyze
from code.lib.utils import normalize
import seaborn as sns
import matplotlib.pyplot as plt


def build_model(init_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(32, activation='relu', input_shape=(init_size, )))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(init_size, )))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu'))

    model.add(keras.layers.Dense(2, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def nn(epochs, batch_size, class_weight):
    # print()
    # print('##################')
    # print('epoch: ', epochs)
    # print('batch size: ', batch_size)
    # print('class weight: ', class_weight[1])
    # data, label = get_train_data()
    x = pd.read_csv('../data/processed/train_data.csv')
    data = normalize(x.drop(columns=['小微企业ID', '是否老赖']).values)

    data = np.concatenate((data, pd.get_dummies(x['是否老赖']).values), axis=1)

    msk = np.random.random(len(data)) < 0.8

    train_data = data[msk]
    eval_data = data[~msk]

    train_class_0 = train_data[np.where(train_data[:, -1] == 0)[0]]
    train_class_1 = train_data[np.where(train_data[:, -1] == 1)[0]]

    down_rate = np.random.random(len(train_class_0)) < 0.5

    train_class_0 = train_class_0[down_rate]

    train_data = train_class_0
    for i in range(5):
        train_data = np.concatenate((train_data, train_class_1), axis=0)
    np.random.shuffle(train_data)

    train_label = train_data[:, -2:]
    train_data = train_data[:, 1:-2]

    eval_label = eval_data[:, -2:]
    eval_data = eval_data[:, 1:-2]

    # print(train_data.shape, train_label.shape, eval_data.shape, eval_label.shape)

    model = build_model(train_data.shape[1])
    # model.summary()

    history = model.fit(train_data, train_label, epochs=epochs,
                        batch_size=batch_size,
                        validation_split=0.2, verbose=0,
                        # 调整loss-weight来解决样本量不平均的问题
                        class_weight=class_weight)
    plot_history(history)

    test_loss, test_acc = model.evaluate(eval_data, eval_label, verbose=0)
    #
    # print('Test loss: ', test_loss)
    # print('Test accuracy: ', test_acc)

    predictions = model.predict(eval_data)
    # result_analyze(eval_data, eval_label, predictions)

    try:
        fpr, tpr, thresholds = metrics.roc_curve(eval_label[:, -1], predictions[:, -1])
    except:
        return
    # print(fpr)
    # print(tpr)
    auc = metrics.auc(fpr, tpr)
    # print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)
    print("epoch: %5d, batch size: %5d, class weight: %5d, loss: %8.4f, acc: %8.4f, auc: %8.4f" %
          (epochs, batch_size, class_weight[1], test_loss, test_acc, auc))

    # print(np.count_nonzero(eval_label[:, -1]))
    # print(np.count_nonzero(np.argmax(predictions, axis=1)))
    cm = metrics.confusion_matrix(eval_label[:, -1], np.argmax(predictions, axis=1))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.show()

    # test_data, test_ids = get_test_data()
    #
    # predictions = model.predict(test_data)
    #
    # df = pd.DataFrame(data={'EID': test_ids, 'FORTARGET': np.argmax(predictions, axis=1), 'PROB': predictions[:,1]})
    # df.to_csv('../data/credit_evaluation.csv', index=False)


if __name__ == '__main__':
    nn(50, 100, {0:1, 1:1})
    exit()
    epoch_list = [10, 20, 30, 40, 50]
    batch_size_list = [100, 120, 140, 160, 180, 200]
    class_weight_list = [
        {0: 1, 1: 1},
        {0: 1, 1: 2},
        {0: 1, 1: 3},
        {0: 1, 1: 4},
        {0: 1, 1: 5},
    ]
    for e in epoch_list:
        for b in batch_size_list:
            for c in class_weight_list:
                nn(e, b, c)
