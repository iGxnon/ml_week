import numpy as np


class GaussianMixtureClusterEM:
    def gaussian(self, X, mu, cov):
        n = X.shape[1]  # dimension
        diff = (X - mu).T  # difference between X and mean vector mu
        gaussian_prob = np.diagonal(1 / ((2 * np.pi) ** (n / 2) * \
                                         np.linalg.det(cov) ** 0.5) \
                                    * np.exp(-0.5 * \
                                             np.dot(np.dot(diff.T, \
                                                           np.linalg.inv(cov)), \
                                                    diff))).reshape(-1, 1)
        return gaussian_prob

    def init_clusters(self, X, n_clusters):
        from sklearn.cluster import KMeans
        clusters = []
        kmeans = KMeans().fit(X)
        mu_k = kmeans.cluster_centers_
        for i in range(n_clusters):
            clusters.append({
                'alpha_k': 1.0 / n_clusters,
                'mu_k': mu_k[i],
                'cov_k': np.identity(X.shape[1], dtype=np.float64)
            })
        return clusters

    def expectation_step(self, X, clusters):
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)
        for cluster in clusters:
            alpha_k = cluster['alpha_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']
            gamma_nk = (alpha_k * self.gaussian(X, mu_k, cov_k)).astype(np.float64)
            for i in range(X.shape[0]):
                totals[i] += gamma_nk[i]  # calculate denominator of Eq.(6.18)
            cluster['gamma_nk'] = gamma_nk
            cluster['totals'] = totals
        for cluster in clusters:
            cluster['gamma_nk'] /= cluster['totals']  # calculate gamma,see Eq.(6.18)

    def maximization_step(self, X, clusters):
        N = float(X.shape[0])
        for cluster in clusters:
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((X.shape[1], X.shape[1]))
            N_k = np.sum(gamma_nk, axis=0)
            alpha_k = N_k / N  # see Eq.(6.22)
            mu_k = np.sum(gamma_nk * X, axis=0) / N_k  # see Eq.(6.19)
            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)
            cov_k /= N_k  # see Eq.(6.20)
            cluster['alpha_k'] = alpha_k
            cluster['mu_k'] = mu_k
            cluster['cov_k'] = cov_k

    def get_likelihood(self, X, clusters):
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
        return np.sum(sample_likelihoods), sample_likelihoods

    def fit(self, X, n_clusters, n_epochs):
        clusters = self.init_clusters(X, n_clusters)
        likelihoods = np.zeros((n_epochs,))
        scores = np.zeros((X.shape[0], n_clusters))
        history = []
        for i in range(n_epochs):
            clusters_snapshot = []
            for cluster in clusters:
                clusters_snapshot.append({'mu_k': cluster['mu_k'].copy(), 'cov_k': cluster['cov_k'].copy()})
            history.append(clusters_snapshot)
            self.expectation_step(X, clusters)
            self.maximization_step(X, clusters)
            likelihood, sample_likelihoods = self.get_likelihood(X, clusters)
            likelihoods[i] = likelihood
            print('Epoch: ', i + 1, 'Likelihood: ', likelihood)
        for i, cluster in enumerate(clusters):
            scores[:, i] = np.log(cluster['gamma_nk']).reshape(-1)
        return clusters, likelihoods, scores, sample_likelihoods, history


from sklearn import datasets

iris = datasets.load_iris()
iris_x = iris.data
iris_label = iris.target
# just using the third and fourth feature
X = iris_x[:, [2, 3]]
# samples of calss0
X_0 = []
for i in range(len(iris_label)):
    if iris_label[i] == 0:
        X_0.append(X[i])
X_0 = np.array(X_0)
# samples of class1
X_1 = []
for i in range(len(iris_label)):
    if iris_label[i] == 1:
        X_1.append(X[i])
X_1 = np.array(X_1)
# samples of class2
X_2 = []
for i in range(len(iris_label)):
    if iris_label[i] == 2:
        X_2.append(X[i])
X_2 = np.array(X_2)
# show original data
import matplotlib.pyplot as plt

plt.scatter(X_0[:, 0], X_0[:, 1], label="class0", marker="o")
plt.scatter(X_1[:, 0], X_1[:, 1], label="class1", marker="x")
plt.scatter(X_2[:, 0], X_2[:, 1], label="class2", marker="+")
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("original sample distribution")
plt.legend()
plt.show()

model = GaussianMixtureClusterEM()
clusters, likelihoods, scores, sample_likelihoods, history = model.fit(X, n_clusters=3, n_epochs=500)
labels = []
for i in range(scores.shape[0]):
    max_value = scores[i, 0]
    max_index = 0
    for j in range(scores.shape[1]):
        if scores[i, j] > max_value:
            max_value = scores[i, j]
            max_index = j
    labels.append(max_index)

plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')
plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("GaussianMixtureClusterEM Clustering")
plt.legend()
plt.show()

from sklearn.mixture import GaussianMixture as GMM

gmm = GMM(n_components=3).fit(X)
labels = gmm.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, s=50, cmap='viridis')

plt.xlabel("petal length")
plt.ylabel("petal width")
plt.title("sklearn GMM Clustering")
plt.legend()
plt.show()
