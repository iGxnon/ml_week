# import necessary libs
import random

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:

    def predict(self, x):
        y = x.dot(self.w) + self.b
        return y

    def loss_function(self, train_x, train_y):
        inst_num = train_x.shape[0]
        pred_y = train_x.dot(self.w) + self.b
        loss = np.sum((pred_y - train_y) ** 2) / (2 * inst_num)
        return loss

    def calculate_grad(self, train_x, train_y):
        inst_num = train_x.shape[0]  # data size
        pred_y = train_x.dot(self.w) + self.b  # training prediction
        grad_w = train_x.T.dot((pred_y - train_y)) / inst_num  # see Eq.(2.5)
        grad_b = np.sum((pred_y - train_y)) / inst_num  # see Eq.(2.6)
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

    def fit(self, train_x, train_y, learn_rate, max_iter, epsilon):

        feat_num = train_x.shape[1]  # feature dimension
        self.w = np.zeros((feat_num, 1))  # initialize model parameters
        self.b = 0.0
        # learn model parameters using gradient descent algorithm
        loss_list = self.gradient_descent(train_x, train_y, learn_rate, max_iter
                                          , epsilon)
        self.training_visualization(loss_list)

    def training_visualization(self, loss_list):
        plt.plot(loss_list, color='red')
        plt.xlabel("iterations")
        plt.ylabel("loss")
        plt.savefig("loss.png", bbox_inches='tight', dpi=400)
        plt.show()


if __name__ == '__main__':
    X = np.linspace(-1, 1, 200)
    Y = 2 * X + np.random.randn(200) * 0.3
    train_x = X.reshape(-1, 1)
    train_y = Y.reshape(-1, 1)
    LR = LinearRegression()
    LR.fit_batch(train_x, train_y, 0.01, 1000, 0.00001)
    plt.plot(X, Y, 'ro', label = "trainning data")
    plt.legend()
    plt.plot(X, LR.w[0, 0] * X + LR.b, ls="-", lw=2, c="b")
    plt.xlabel("x")
    plt.ylabel("y")
    s = "y=%.3f*x%.3f" % (LR.w[0, 0], LR.b)
    plt.text(0, LR.b - 0.2, s, color="b")
    plt.savefig("result.png", bbox_inches='tight', dpi = 400)
    plt.show()