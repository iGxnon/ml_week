from random import randrange
from random import randint
from sklearn.tree import DecisionTreeClassifier
import numpy as np


class RandomForest:
    # boost-trap
    def boosttrap_sampling(self, data_length):
        sample_data_index = []
        while len(sample_data_index) < data_length:
            index = randrange(data_length - 1)
            sample_data_index.append(index)
        return sample_data_index

    def random_select_k_features(self, feature_length, k):
        feature_index = []
        while len(feature_index) < k:
            index = randint(0, feature_length - 1)
            if index in feature_index:
                index = randint(0, feature_length - 1)
            else:
                feature_index.append(index)
        return feature_index

    def get_sampled_data(self, data_x, data_y, k):
        data_len = data_x.shape[0]
        feat_len = data_x.shape[1]
        sample_data_index = self.boosttrap_sampling(data_len)
        feature_index = self.random_select_k_features(feat_len, k)
        sample_data_x = data_x[sample_data_index]
        sample_data_x = sample_data_x[:, feature_index]
        sample_data_y = data_y[sample_data_index]
        return sample_data_x, sample_data_y, feature_index

    def fit(self, train_x, train_y, tree_num, k, tree_depth):
        self.feature_list = []
        self.trees = []
        for i in range(tree_num):
            sample_data_x, sample_data_y, feature_index = self.get_sampled_data(
                train_x, train_y, k)
            self.feature_list.append(feature_index)
            clf = DecisionTreeClassifier(criterion='gini', max_depth=tree_depth)
            clf.fit(sample_data_x, sample_data_y)
            self.trees.append(clf)

    def predict(self, test_x):
        pred_result = np.zeros((len(test_x), len(self.trees)), dtype=int)

        labels = []
        for i in range(len(self.trees)):
            test_x_sub = test_x[:, self.feature_list[i]]
            pred_y = self.trees[i].predict(test_x_sub)
            pred_result[:, i] = pred_y
        for i in range(len(test_x)):
            label = self.majorityCount(pred_result[i, :])
            labels.append(label)
        return pred_result, labels

    def majorityCount(self, votes):
        class_list = []
        for c in votes:
            if c not in class_list:
                class_list.append(c)
        count = []
        for c in class_list:
            num = 0
            for x in votes:
                if x == c:
                    num += 1
            count.append(num)
        max_count = 0
        max_index = 0
        for i in range(len(count)):
            if count[i] > max_count:
                max_count = count[i]
                max_index = i
        return class_list[max_index]


if __name__ == '__main__':
    import pandas as pd

    data = pd.read_csv("Credit_Card_Sale.csv")
    data = data.values
    data_x = data[:, 0:-1]  # sample feature
    data_y = data[:, -1]  # sample label
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)
    rf_model = RandomForest()
    rf_model.fit(X_train, y_train, 50, 4, 3)
    _, test_y_rf = rf_model.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == test_y_rf[i]:
            acc += 1.0
    print("RF model accuracy:%.3f" % (acc / len(y_test)))
    from sklearn.ensemble import RandomForestClassifier

    rf_sklearn = RandomForestClassifier(max_depth=3, n_estimators=50)
    rf_sklearn.fit(X_train, y_train)
    y_pred = rf_sklearn.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == y_pred[i]:
            acc += 1.0
    print("RF Sklearn model accuracy:%.3f" % (acc / len(y_test)))
    tree_model = DecisionTreeClassifier(max_depth=3)
    tree_model.fit(X_train, y_train)
    tree_model_test_y = tree_model.predict(X_test)
    acc = 0.0
    for i in range(len(y_test)):
        if y_test[i] == tree_model_test_y[i]:
            acc += 1.0
    print("single tree model accuracy:%.3f" % (acc / len(y_test)))
