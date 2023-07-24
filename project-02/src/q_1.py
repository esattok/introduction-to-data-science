import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
from scipy.linalg import eigh

# Load dataset
data = loadmat("./digits/digits.mat")
params = data["digits"]
labels = data["labels"]

# Creating train and test datasets
train_p, test_p, train_l, test_l = train_test_split(params, labels, test_size=0.5, shuffle=True)

# Question 1.1
centered = train_p - np.mean(train_p, axis=0)
cov_matrix = np.cov(centered, rowvar=False)
print("Shape of variance matrix =", cov_matrix.shape)

values, vectors = eigh(cov_matrix)
sorted_values = np.argsort(values)[::-1]
sorted_vectors = vectors[:, sorted_values]

pca = PCA(n_components=400)
pca.fit(centered)
percentage_var_explained = values[sorted_values] / np.sum(values)
cum_var_explained = np.cumsum(percentage_var_explained)

# Question 1.1
plt.plot(percentage_var_explained, linewidth=2)
plt.title("Eigen Values Explained by Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Eigen Values")
plt.grid(True)
plt.show()

# Question 1.3
mean_picture_pca = plt.imshow((pca.mean_).reshape(20, 20))
plt.title("Mean of Training Data")
plt.axis("off")
plt.colorbar()
plt.show()

# Question 1.3
fig, axes = plt.subplots(nrows=8, ncols=8, figsize=(8, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(20, 20), cmap='gray')
    ax.axis("off")
plt.suptitle("First 64 Principal Components")
plt.tight_layout()
plt.show()

# Question 1.4
train_errors = []
test_errors = []
for i in range(1, 101):
    pca = sorted_vectors[:, :i]
    train_centered = train_p - np.mean(train_p, axis=0)
    train_X = np.dot(train_centered, pca)
    test_centered = test_p - np.mean(train_p, axis=0)
    test_X = np.dot(test_centered, pca)
    classifier = GaussianNB()
    classifier.fit(train_X, train_l.ravel())
    train_pred = classifier.predict(train_X)
    train_errors.append(1 - accuracy_score(train_l, train_pred))
    test_pred = classifier.predict(test_X)
    test_errors.append(1 - accuracy_score(test_l, test_pred))

# Question 1.5
plt.plot(range(1, 101), train_errors, label="Training Error")
plt.plot(range(1, 101), test_errors, label="Testing Error")
plt.title("Classification Error vs Number of Principal Components")
plt.xlabel("Number of Principal Components")
plt.ylabel("Classification Error")
plt.legend()
plt.grid(True)
plt.show()

# Separating the lines into two plots
# Plot the train statistics
plt.plot(range(1, 101), train_errors)
plt.title("Training Result Statistic")
plt.xlabel("Number of Principal Components")
plt.ylabel("Classification Error")
plt.grid(True)
plt.show()

# Plot the test statistics
plt.plot(range(1, 101), test_errors)
plt.title("Testing Result Statistic")
plt.xlabel("Number of Principal Components")
plt.ylabel("Classification Error")
plt.grid(True)
plt.show()