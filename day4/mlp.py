import numpy as np


class NeuralNetwork:

    def __init__(self, layer_sizes):

        self.num_layers = len(layer_sizes)  # layer number of NN
        self.layers = layer_sizes  # node numbers of each layer
        self.weights = [np.random.randn(y, x) for x, y in zip(layer_sizes[:-1],
                                                              layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    @staticmethod
    def sigmoid(z):
        act = 1.0 / (1.0 + np.exp(-z))
        return act

    @staticmethod
    def sigmoid_prime(z):
        act = NeuralNetwork.sigmoid(z) * (1.0 - NeuralNetwork.sigmoid(z))
        return act

    @staticmethod
    def tanh(z):
        return 2 * NeuralNetwork.sigmoid(2 * z) - 1

    @staticmethod
    def tanh_prime(z):
        return 4 * NeuralNetwork.sigmoid_prime(2 * z)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_prime(z):
        return np.where(z > 0, 1, 0)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.maximum(alpha * z, z)

    @staticmethod
    def leaky_relu_prime(z, alpha=0.01):
        return np.where(z > 0, 1, alpha)

    def feed_forward(self, x):
        output = x.copy()
        for w, b in zip(self.weights, self.biases):
            output = self.tanh(np.dot(w, output) + b)
        return output

    def feed_backward(self, x, y):
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]
        activation = np.transpose(x)
        activations = [activation]
        layer_input = []
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            layer_input.append(z)  # input of each layer
            activation = self.tanh(z)
            activations.append(activation)  # output of each layer

        # loss function
        ground_truth = np.transpose(y)
        diff = activations[-1] - ground_truth
        # get input of last layer
        last_layer_input = layer_input[-1]
        delta = np.multiply(diff, self.tanh_prime(last_layer_input))
        # bias update of last layer
        delta_b[-1] = np.sum(delta, axis=1, keepdims=True)
        # weight update of last layer
        delta_w[-1] = np.dot(delta, np.transpose(activations[-2]))
        # update weights and bias from 2nd layer to last layer
        for i in range(2, self.num_layers):
            input_values = layer_input[-i]
            delta = np.multiply(np.dot(np.transpose(self.weights[-i + 1]), delta), self.tanh_prime(input_values))
            delta_b[-i] = np.sum(delta, axis=1, keepdims=True)
            delta_w[-i] = np.dot(delta, np.transpose(activations[-i - 1]))
        return delta_b, delta_w

    def fit(self, x, y, lr, mini_batch_size, epochs=1000):
        n = len(x)  # training size
        for i in range(epochs):
            print("=" * 10 + f" Epoch {i + 1} " + "=" * 10)
            random_list = np.random.randint(0, n - mini_batch_size, int(n / mini_batch_size))
            batch_x = [x[k:k + mini_batch_size] for k in random_list]
            batch_y = [y[k:k + mini_batch_size] for k in random_list]
            for j in range(len(batch_x)):
                delta_b, delta_w = self.feed_backward(batch_x[j], batch_y[j])
                self.weights = [w - (lr / mini_batch_size) * dw for w, dw in
                                zip(self.weights, delta_w)]
                self.biases = [b - (lr / mini_batch_size) * db for b, db in
                               zip(self.biases, delta_b)]

            labels = self.predict(x)
            acc = 0.0
            for k in range(len(labels)):
                if y[k, labels[k]] == 1.0:
                    acc += 1.0
            acc = acc / len(labels)
            print("epoch train %d accuracy %.3f" % (i + 1, acc))

    def predict(self, x):
        results = self.feed_forward(x.T)
        labels = [np.argmax(results[:, y]) for y in range(results.shape[1])]
        return labels
