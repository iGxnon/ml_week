import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from Support_Vector_Machine import SupportVectorMachine
import numpy as np
import pandas as pd
from sklearn.svm import SVC

data, label = make_blobs(n_samples=200, n_features=2, centers=2)
for i in range(len(label)):
    if label[i] == 0:
        label[i] = -1
df = pd.DataFrame()
df['x1'] = data[:, 0]
df['x2'] = data[:, 1]
df['class'] = label
positive = df[df["class"] == 1]
negative = df[df["class"] == -1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive["x1"], positive["x2"], s=30, c="b", marker="o", label="class 1")
ax.scatter(negative["x1"], negative["x2"], s=30, c="r", marker="x", label="class -1")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()
orig_data = df.values
cols = orig_data.shape[1]
data_mat = orig_data[:, 0:cols - 1]
label_mat = orig_data[:, cols - 1:cols]
model = SupportVectorMachine(data_mat, label_mat, 0.6, 0.001, 100)
model.smo()
support_x = []
support_y = []
class1_x = []
class1_y = []
class01_x = []
class01_y = []
for i in range(200):
    if model.alphas[i] > 0.0:
        support_x.append(data_mat[i, 0])
        support_y.append(data_mat[i, 1])
for i in range(200):
    if label_mat[i] == 1:
        class1_x.append(data_mat[i, 0])
        class1_y.append(data_mat[i, 1])
    else:
        class01_x.append(data_mat[i, 0])
        class01_y.append(data_mat[i, 1])
w_best = np.dot(np.multiply(model.alphas, label_mat).T, data_mat)
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(support_x, support_y, s=100, c="y", marker="v", label="support_v")
ax.scatter(class1_x, class1_y, s=30, c="b", marker="o", label="class 1")
ax.scatter(class01_x, class01_y, s=30, c="r", marker="x", label="class -1")
a = min(data_mat[:, 0])
b = max(data_mat[:, 0])
lin_x = np.linspace(a, b, 200)
lin_y = (-float(model.b) - w_best[0, 0] * lin_x) / w_best[0, 1]
plt.plot(lin_x, lin_y, color="green")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()


svc_model = SVC(kernel='linear')
svc_model.fit(data_mat, label_mat)
class1_x = []
class1_y = []
class01_x = []
class01_y = []
for i in range(200):
    if label_mat[i] == 1:
        class1_x.append(data_mat[i, 0])
        class1_y.append(data_mat[i, 1])
    else:
        class01_x.append(data_mat[i, 0])
        class01_y.append(data_mat[i, 1])
w_best = svc_model.coef_
model_b = svc_model.intercept_[0]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(svc_model.support_vectors_[:, 0], svc_model.support_vectors_[:, 1], s=100, c="y", marker="v",
           label="support_v")
ax.scatter(class1_x, class1_y, s=30, c="b", marker="o", label="class 1")
ax.scatter(class01_x, class01_y, s=30, c="r", marker="x", label="class -1")
a = min(data_mat[:, 0])
b = max(data_mat[:, 0])
lin_x = np.linspace(a, b, 200)
lin_y = (-float(model_b) - w_best[0, 0] * lin_x) / w_best[0, 1]
plt.plot(lin_x, lin_y, color="green")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()
