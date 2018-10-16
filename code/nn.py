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
    model.add(keras.layers.Dense(64, activation='relu', input_shape=(init_size, )))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(16, activation='relu'))

    model.add(keras.layers.Dense(2, activation=tf.nn.sigmoid))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


EPOCHS = 20


def nn():
    # data, label = get_train_data()
    x = pd.read_csv('../data/processed/train_data.csv')

    drop_list = [
    '行业门类代码',
    '投资总额(万元)',
    '欠税税务机关',
    '行政处罚次数',
    '处罚机关_市场',
    '处罚机关_邮政',
    '审理机关_6_x',
    '审理机关_7_x',
    '文书类型_3.0',
    '文书类型_8.0',
    '诉讼地位_10',
    '审理机关_0_x',
    '诉讼地位_15',
    '审理程序_1',
    '审理机关_1_x',
    '审理机关_2_x',
    '公告类型_1',
    '公告类型_2',
    '公告类型_3',
    '公告类型_12',
    '公告类型_13',
    '公告类型_14',
    '公告类型_15',
    '公告类型_16',
    '公告类型_17',
    '公告类型_18',
    '公告类型_19',
    '公告类型_21',
    '公告类型_22',
    '公告类型_23',
    '公告类型_25',
    '诉讼地位3_6',
    '诉讼地位3_7'
    ]
    # x = x.drop(drop_list, axis=1)  # do not modify x, we will use it later

    label = pd.get_dummies(x['是否老赖']).values
    data = x.drop(columns=['小微企业ID', '是否老赖'])
    data = normalize(data.values)

    msk = np.random.random(len(data)) < 0.8
    train_label = label[msk]
    train_data = data[msk]

    eval_label = label[~msk]
    eval_data = data[~msk]

    print(train_data.shape, eval_data.shape)

    model = build_model(train_data.shape[1])
    model.summary()

    history = model.fit(train_data, train_label, epochs=EPOCHS, batch_size=100,
                        validation_split=0.2, verbose=0)
    plot_history(history)

    test_loss, test_acc = model.evaluate(eval_data, eval_label)

    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(eval_data)
    # result_analyze(eval_data, eval_label, predictions)

    fpr, tpr, thresholds = metrics.roc_curve(eval_label[:, -1], predictions[:, -1])
    # print(fpr)
    # print(tpr)
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    # plot_roc(fpr, tpr, auc)

    print(np.count_nonzero(eval_label[:, -1]))
    print(np.count_nonzero(np.argmax(predictions, axis=1)))
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
    nn()
