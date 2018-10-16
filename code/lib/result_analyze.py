import numpy as np


def result_analyze(data, label, predictions):
    print(data)
    print(label)
    print(predictions)
    exit()
    pred = np.argmax(predictions, axis=1)

    # label 为1   预测正确的ID
    tp_ids = []
    # label 为1   预测失败的ID
    fp_ids = []
    # label 为0   预测正确的ID
    tn_ids = []
    # label 为0   预测失败的ID
    fn_ids = []

    for i, r in label:
        if r[1] == 0:
            if pred[i] == 0:
                tn_ids.append(r[0])
            else:
                fn_ids.append(r[0])
        else:
            if pred[i] == 1:
                tp_ids.append(r[0])
            else:
                fp_ids.append(r[0])
    print(tp_ids)
    print(fp_ids)
