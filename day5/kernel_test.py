import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.datasets import make_circles
from Support_Vector_Machine import SupportVectorMachine

data, label = make_circles(n_samples=200, factor=0.5, shuffle=True)

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
data_mat = SupportVectorMachine.map_kernel("radial", data_mat)
model = SupportVectorMachine(data_mat, label_mat, 0.6, 0.001, 100)

df = pd.DataFrame()
df['x1'] = np.array(data_mat[:, 0]).squeeze()
df['x2'] = np.array(data_mat[:, 1]).squeeze()
df['class'] = np.array(model.y_train).squeeze()

positive = df[df["class"] == 1]
negative = df[df["class"] == -1]
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(positive["x1"], positive["x2"], s=30, c="b", marker="o", label="class 1")
ax.scatter(negative["x1"], negative["x2"], s=30, c="r", marker="x", label="class -1")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()
