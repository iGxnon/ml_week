import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.class_prior_probs = {}
        self.class_condt_probs = {}

    def estimate_class_prior_probs(self, y_train):
        inst_num = len(y_train)
        for i in range(inst_num):
            key = y_train[i]
            if key not in self.class_prior_probs.keys():
                self.class_prior_probs[key] = 0
            self.class_prior_probs[key] += 1
        for key in self.class_prior_probs:
            self.class_prior_probs[key] = \
                float(self.class_prior_probs[key]) / inst_num

    def estimate_class_condt_probs(self, x_train, y_train, feature_names, feature_types, labels):
        feature_num = len(feature_names)
        inst_num = len(x_train)
        self.inst_num = inst_num
        for label in labels:
            self.class_condt_probs[label] = {}
            self.class_condt_probs[label]["num"] = 0
            for feature in feature_names:
                self.class_condt_probs[label][feature] = {}
            for i in range(inst_num):
                if y_train[i] == label:
                    self.class_condt_probs[label]["num"] += 1
            for i in range(feature_num):
                if feature_types[i] == 1:  # for discrete feature
                    for j in range(inst_num):
                        if y_train[j] == label:
                            if x_train[j, i] not in self.class_condt_probs[label][feature_names[i]].keys():
                                self.class_condt_probs[label][feature_names[i]][x_train[j, i]] = 0
                            self.class_condt_probs[label][feature_names[i]][x_train[j, i]] += 1
                else:  # for continuous feature
                    values = []
                    for j in range(inst_num):
                        if y_train[j] == label:
                            values.append(float(x_train[j, i]))
                    values = np.array(values)
                    mean_values = np.mean(values)
                    var_values = np.var(values)
                    self.class_condt_probs[label][feature_names[i]]["mean"] = mean_values
                    self.class_condt_probs[label][feature_names[i]]["var"] = var_values

    def normal_prob_dense(self, x, mu, std):
        import math
        p = 1.0 / (std * pow(2 * math.pi, 0.5)) * np.exp(-((x - mu) ** 2) / (2 * std ** 2))
        return p

    def fit(self, x_train, y_train, feature_names, feature_types, labels):
        self.estimate_class_prior_probs(y_train)
        self.estimate_class_condt_probs(x_train, y_train, feature_names, feature_types, labels)

    def predict(self, x_test, feature_names, feature_types):
        m, n = np.shape(x_test)
        pred_labels = []
        for i in range(m):  # for each sample
            union_probs = {}  # calcute P(x,c)
            for label in self.class_condt_probs.keys():
                prob = float(self.class_condt_probs[label]["num"]) / self.inst_num
                for j in range(n):
                    if feature_types[j] == 1:  # for discrete feature, see Eq.(6.10)
                        prob = prob * float(self.class_condt_probs[label] \
                                                [feature_names[j]][x_test[i, j]]) \
                               / self.class_condt_probs[label]["num"]
                    else:  # for continous feature, see Eq.(6.11)
                        mu = float(self.class_condt_probs[label][feature_names[j]]["mean"] + 1e-10)
                        threta = float(self.class_condt_probs[label][feature_names[j]]["var"] + 1e-10)
                        p_condit = self.normal_prob_dense(float(x_test[i, j]), mu, threta ** 0.5)
                        prob = prob * p_condit
                union_probs[label] = prob
            pred_label = None
            max_prob = -1.0
            for key in union_probs.keys():
                if union_probs[key] > max_prob:
                    max_prob = union_probs[key]
                    pred_label = key
            pred_labels.append(pred_label)
        return pred_labels


from sklearn.datasets import load_breast_cancer

cancer_data = load_breast_cancer()
data = cancer_data['data']  # feature space
y = cancer_data['target']  # label space
feature_names = cancer_data['feature_names']
label_names = [0, 1]  # 1 for malignant, 0 for benign
feature_types = [0 for i in range(len(feature_names))]

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.3)
model = NaiveBayesClassifier()
model.fit(X_train, y_train, feature_names, feature_types, label_names)
pred_test = model.predict(X_test, feature_names, feature_types)
acc = 0.0
for i in range(len(y_test)):
    if y_test[i] == pred_test[i]:
        acc += 1.0
print("NaiveBayesClassifier Accuracy:%.3f" % (acc / len(y_test)))

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
pred_test = gnb.predict(X_test)
acc = 0.0
for i in range(len(y_test)):
    if y_test[i] == pred_test[i]:
        acc += 1.0
print("sklearn GaussianNB Accuracy:%.3f" % (acc / len(y_test)))
