import random
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd

warnings.filterwarnings('ignore')


class LogisticRegression:
    def sigmoid(self, x):
        y_prob = 1.0 / (1.0 + np.exp(-x))
        return y_prob

    # define prediction function
    def predict_prob(self, x):
        y_prob = self.sigmoid(np.dot(x, self.w) + self.b)  # see Eq.(2.7)
        return y_prob

    def predict(self, X):
        inst_num = X.shape[0]
        probs = self.predict_prob(X)
        labels = np.zeros(inst_num)
        for i in range(inst_num):
            if probs[i] >= 0.5:
                labels[i] = 1
        return probs, labels

    def loss_function(self, train_x, train_y):
        inst_num = train_x.shape[0]
        loss = 0.0
        for i in range(inst_num):
            z = np.dot(train_x[i, :], self.w) + self.b
            loss += -train_y[i] * z + np.log(1 + np.exp(z))  # see Eq.(2.10)
        loss = loss / inst_num
        return loss

    def calculate_grad(self, train_x, train_y):
        inst_num = train_x.shape[0]  # data size
        probs = self.sigmoid(train_x.dot(self.w) + self.b)  # training prediction

        grad_w = np.sum(np.dot(train_x.T, probs - train_y)) / inst_num
        grad_b = np.sum(probs - train_y) / inst_num

        return grad_w, grad_b

    def gradient_descent(self, train_x, train_y, learn_rate, max_iter, epsilon):
        loss_list = []
        for i in range(max_iter):
            loss_old = self.loss_function(train_x, train_y)
            loss_list.append(loss_old)
            grad_w, grad_b = self.calculate_grad(train_x, train_y)
            self.w = self.w - learn_rate * grad_w
            self.b = self.b - learn_rate * grad_b
            loss_new = self.loss_function(train_x, train_y)
            if abs(loss_new - loss_old) <= epsilon:
                break
        return loss_list

    def fit(self, train_x, train_y, learn_rate, max_iter, epsilon):
        feat_num = train_x.shape[1]  # feature dimension
        self.w = np.zeros((feat_num, 1))  # initialize model parameters
        self.b = 0.0
        # learn model parameters using gradient descent algorithm
        loss_list = self.gradient_descent(train_x, train_y, learn_rate, max_iter
                                          , epsilon)
        self.training_visualization(loss_list)

    @staticmethod
    def batch_loader(X, y, batch_size=16, seed=114514):
        size = X.shape[0]
        indices = list(range(size))
        random.seed(seed)
        random.shuffle(indices)
        for batch_indices in [indices[i:i + batch_size] for i in
                              range(0, size, batch_size)]:
            yield X[batch_indices], y[batch_indices]

    def batch_gradient_descent(self, train_x, train_y, learn_rate, max_iter, epsilon, batch_size=16, seed=114514):
        """
        随机小批量梯度下降
        """
        loss_list = []
        for _ in range(max_iter):
            losses = []
            for batch_x, batch_y in self.batch_loader(train_x, train_y, batch_size, seed):
                losses.extend(self.gradient_descent(batch_x, batch_y, learn_rate, 1, epsilon))
            loss_list.append(np.mean(losses))
            if len(loss_list) > 2 and abs(loss_list[-1] - loss_list[-2]) <= epsilon:
                break

        return loss_list

    def fit_batch(self, train_x, train_y, learn_rate, max_iter, epsilon, batch_size=16, seed=114514):
        feat_num = train_x.shape[1]  # feature dimension
        self.w = np.zeros((feat_num, 1))  # initialize model parameters
        self.b = 0.0
        # learn model parameters using gradient descent algorithm
        loss_list = self.batch_gradient_descent(train_x, train_y, learn_rate, max_iter
                                                , epsilon, batch_size, seed)
        self.training_visualization(loss_list)

    def training_visualization(self, loss_list):
        plt.plot(loss_list, color='red')
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.savefig("loss.png", bbox_inches='tight', dpi=400)
        plt.show()


def test():
    from sklearn.datasets import make_blobs
    # make blob data
    data, label = make_blobs(n_samples=200, n_features=2, centers=2)
    train_x = np.array(data)
    label = np.array(label)
    train_label = label.reshape(-1, 1)
    # train logistic regression model
    LR = LogisticRegression()
    LR.fit(data, train_label, 0.01, 1000, 0.00001)


    df = pd.DataFrame()
    df['x1'] = data[:, 0]
    df['x2'] = data[:, 1]
    df['class'] = label

    positive = df[df["class"] == 1]
    negative = df[df["class"] == 0]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(positive["x1"], positive["x2"], s=30, c="b", marker="o", label="class 1")
    ax.scatter(negative["x1"], negative["x2"], s=30, c="r", marker="x", label="class 0")
    ax.legend()
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    orig_data = df.values
    cols = orig_data.shape[1]
    data_mat = orig_data[:, 0:cols - 1]
    a = min(data_mat[:, 0])
    b = max(data_mat[:, 0])
    lin_x = np.linspace(a, b, 200)
    lin_y = (-float(LR.b) - LR.w[0, 0] * lin_x) / LR.w[1, 0]
    plt.plot(lin_x, lin_y, color="red")
    plt.savefig("result.png", bbox_inches='tight', dpi=400)


if __name__ == '__main__':
    test()