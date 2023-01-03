import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from logistic_regression import LogisticRegression as LR
from sklearn.linear_model import LogisticRegression as LR_SKL

if __name__ == '__main__':
    f = open('Stock_Client_loss.csv')
    data = pd.read_csv(f)
    data_x = data[["x1", "x2", "x3", "x4", "x5"]]
    data_y = np.array(data["loss"])
    # data normalization
    scaler = StandardScaler()
    data_x = scaler.fit_transform(data_x)
    # divide data into train/test, 70% for train, 30% for test
    X_train, X_test, Y_train, Y_test = train_test_split(data_x,
                                                        data_y,
                                                        test_size=0.3,
                                                        shuffle=True)
    learnrate = 0.03
    maxiter = 1200
    eps = 1e-5


    def cal_acc(y_test, y_pred):
        acc = 0.0
        for i in range(len(y_test)):
            if y_test[i] == y_pred[i]:
                acc += 1.0
        return acc / len(y_test)


    LR_model = LR()
    LR_model.fit_batch(train_x=X_train, train_y=Y_train.reshape(-1, 1),
                       learn_rate=learnrate, max_iter=maxiter, epsilon=eps)
    _, Y_test_pred_LR = LR_model.predict(X_test)
    acc = cal_acc(Y_test, Y_test_pred_LR)
    print("LR ACC:%.3f" % (acc))

    LR_SKL_model = LR_SKL()
    LR_SKL_model.fit(X_train, Y_train)
    Y_test_pred_LR_SKL = LR_SKL_model.predict(X_test)
    acc = cal_acc(Y_test, Y_test_pred_LR_SKL)
    print("LR_SKL ACC:%.3f" % (acc))
