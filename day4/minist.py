import struct
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from day4.mlp import NeuralNetwork


def load_mnist_data(directory, kind='train'):
    label_path = os.path.join(os.getcwd(), directory, '%s-labels.idx1-ubyte' % kind)
    image_path = os.path.join(os.getcwd(), directory, '%s-images.idx3-ubyte' % kind)
    with open(label_path, 'rb') as lbpath:  # open label file
        struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(image_path, 'rb') as imgpath:  # open image file
        struct.unpack('>IIII', imgpath.read(16))
        # transform image into 784-dimensional feature vector
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def show_image(image):
    plt.figure()
    img = image.reshape(28, 28)
    plt.imshow(img, 'gray')
    plt.show()


path = 'MNIST'
train_images, train_labels = load_mnist_data(path, kind='train')
train_y = np.zeros((len(train_labels), 10))
for i in range(len(train_labels)):
    train_y[i, train_labels[i]] = 1
scaler = StandardScaler()
train_x = scaler.fit_transform(train_images)
test_images, test_labels = load_mnist_data(path, kind='t10k')
test_y = np.zeros((len(test_labels), 10))
for i in range(len(test_labels)):
    test_y[i, test_labels[i]] = 1
test_x = scaler.fit_transform(test_images)

print("=" * 10 + " Information " + "=" * 10)
print(f"Train data size: {len(train_x)}, shape: {train_x.shape}")
print(f"Test data size: {len(test_x)}, shape: {test_x.shape}")
print("=" * 30)
print()


def sklearn(activation="logistic", trainer="sgd"):
    model = MLPClassifier(hidden_layer_sizes=100, activation=activation, solver=trainer, batch_size=100,
                          learning_rate='constant',
                          learning_rate_init=0.1, max_iter=1000, random_state=0)
    model.fit(train_x, train_y)
    labels = model.predict(test_x)
    acc = 0.0
    for k in range(len(labels)):
        index = 0
        for j in range(10):
            if labels[k, j] == 1:
                index = j
                break
        if test_y[k, index] == 1.0:
            acc += 1.0
    acc = acc / len(labels)
    print("sklearn test accuracy: %.3f" % acc)


def mlp():
    layer_sizes = [784, 100, 10]
    NN = NeuralNetwork(layer_sizes)
    NN.fit(train_x, train_y, lr=0.1, mini_batch_size=100, epochs=1000)
    test_pred_labels = NN.predict(test_x)

    acc = 0.0
    for k in range(len(test_pred_labels)):
        if test_y[k, test_pred_labels[k]] == 1.0:
            acc += 1.0
    acc = acc / len(test_pred_labels)
    print("test accuracy:%.3f" % acc)


if __name__ == '__main__':
    mlp()
