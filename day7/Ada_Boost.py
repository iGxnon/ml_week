import numpy as np
from sklearn.tree import DecisionTreeClassifier


class AdaBoost:
    def fit(self, train_x, train_y, clf_num):
        self.weak_clfs = []
        self.clf_alphas = []
        n_train = len(train_x)  # train size
        w = np.ones(n_train) / n_train  # initial sample weights
        for i in range(clf_num):
            clf = DecisionTreeClassifier(max_depth=3)
            clf.fit(train_x, train_y, sample_weight=w)
            self.weak_clfs.append(clf)
            pred_train_i = clf.predict(train_x)
            error = [int(x) for x in (pred_train_i != train_y)]
            # print("the %d th weak classifier accuracy:%.3f"%(i+1,1-sum(error)/n_train))
            err_wighted = np.dot(w, error)
            # calculate alpha_i
            alpha_i = 0.5 * np.log((1 - err_wighted) / (err_wighted))
            self.clf_alphas.append(alpha_i)
            # update sample weights
            miss = [x if x == 1 else -1 for x in error]
            w = np.multiply(w, np.exp([float(x) * alpha_i for x in miss]))
            w = w / sum(w)

    def predict(self, test_x):
        n_test = len(test_x)

        pred_test = np.zeros(n_test)
        for i in range(len(self.weak_clfs)):
            pred_test_i = self.weak_clfs[i].predict(test_x)
            pred_test_i = [1 if x == 1 else -1 for x in pred_test_i]
            pred_test = pred_test + np.multiply(self.clf_alphas[i], pred_test_i)
        pred_test = (pred_test > 0) * 1
        return pred_test


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv("Credit_Card_Sale.csv")
    data = data.values
    data_x = data[:, 0:-1]  # sample feature
    data_y = data[:, -1]  # sample label
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
    ada_boost = AdaBoost()
    ada_boost.fit(X_train, y_train, clf_num=20)
    test_y_ada_boost = ada_boost.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == test_y_ada_boost[i]:
            acc += 1.0
    print("AdaBoost model accuracy:%.3f" % (acc / len(y_test)))
    from sklearn.ensemble import AdaBoostClassifier

    ada_boost_sklearn = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3), n_estimators=20)
    ada_boost_sklearn.fit(X_train, y_train)
    y_pred = ada_boost_sklearn.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            acc += 1.0
    print("AdaBoost Sklearn model accuracy:%.3f" % (acc / len(y_test)))
    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(X_train, y_train)
    tree_model_test_y = tree_model.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == tree_model_test_y[i]:
            acc += 1.0
    print("single tree model accuracy:%.3f" % (acc / len(y_test)))
