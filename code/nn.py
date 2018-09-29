import tensorflow.keras as keras
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from code.data_process import get_train_data
from code.data_process import get_test_data
from sklearn import metrics
from code.lib.plot import plot_history, plot_roc


def build_model(init_size):
    model = keras.Sequential()
    model.add(keras.layers.Dense(128, activation='relu', input_shape=(init_size, )))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(64, activation='relu'))
    model.add(keras.layers.Dropout(0.5))
    model.add(keras.layers.Dense(32, activation='relu'))
    model.add(keras.layers.Dropout(0.5))

    model.add(keras.layers.Dense(2, activation=tf.nn.softmax))

    model.compile(optimizer=tf.train.AdamOptimizer(0.001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


EPOCHS = 30


def nn():
    data, label = get_train_data()
    msk = np.random.random(len(data)) < 0.8
    train_label = label[msk]
    train_data = data[msk]

    test_label = label[~msk]
    test_data = data[~msk]

    print(train_data.shape, test_data.shape)

    model = build_model(train_data.shape[1])
    # model.summary()

    history = model.fit(train_data, train_label, epochs=EPOCHS,batch_size=1000,
                        validation_split=0.2, verbose=0)
    # plot_history(history)

    test_loss, test_acc = model.evaluate(test_data, test_label)

    print('Test loss: ', test_loss)
    print('Test accuracy: ', test_acc)

    predictions = model.predict(test_data)

    fpr, tpr, thresholds = metrics.roc_curve(test_label[:, -1], predictions[:, -1])
    auc = metrics.auc(fpr, tpr)
    print("Test Auc: ", auc)
    plot_roc(fpr, tpr, auc)

    test_data, test_ids = get_test_data()

    predictions = model.predict(test_data)

    df = pd.DataFrame(data={'EID': test_ids, 'FORTARGET': np.argmax(predictions, axis=1), 'PROB': predictions[:,1]})
    df.to_csv('../data/credit_evaluation.csv', index=False)


if __name__ == '__main__':
    nn()
