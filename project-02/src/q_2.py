import numpy as np
from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.manifold import Isomap
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def read_mnist_images(filename):
    with open(filename, 'r') as f:
        data = np.loadtxt(f)
        data = data.reshape(-1, 400) # flatten the images into 1D arrays
        return data

def read_mnist_labels(filename):
    return np.loadtxt(filename)


# Load the MNIST dataset
images_data = read_mnist_images('./digits/digits.txt')
labels_data = read_mnist_labels('./digits/labels.txt')

# Question 2.1
# Create an instance of Isomap with the desired number of components
isomap = manifold.Isomap(n_components=2)

# Fit the Isomap model to the full dataset
embedded_train_images = isomap.fit_transform(images_data)

# Plot the embeddings, color-coded by the digit labels
plt.scatter(embedded_train_images[:, 0], embedded_train_images[:, 1], c=labels_data, cmap='jet')
plt.colorbar()
plt.title('Isomap Embeddings of MNIST Dataset')
plt.show()

# Question 2.3
# Split the data into training and testing sets
train_images, test_images, train_labels, test_labels = train_test_split(images_data, labels_data, test_size=0.5, random_state=42)

# Define a list of dimensions to test
dimensions = [i for i in range(1, 201, 10)]

# Train a Gaussian classifier for each dimensionality and compute the classification error
train_errors = []
test_errors = []
for dim in dimensions:
    # Reduce the dimensionality of the data using Isomap
    isomap = Isomap(n_components=dim)
    X_train_iso = isomap.fit_transform(train_images)
    X_test_iso = isomap.transform(test_images)

    # Train the classifier on the reduced data
    clf = GaussianNB()
    clf.fit(X_train_iso, train_labels)

    # Test the classifier on the training set
    y_train_pred = clf.predict(X_train_iso)
    train_errors.append(1 - accuracy_score(train_labels, y_train_pred))

    # Test the classifier on the test set
    y_test_pred = clf.predict(X_test_iso)
    test_errors.append(1 - accuracy_score(test_labels, y_test_pred))

# Plot classification error vs. dimension
# Plot classification error vs. dimension
plt.plot(dimensions, train_errors, label='Training Error')
plt.plot(dimensions, test_errors, label='Test Error')
plt.title('Classification Error vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Classification Error')
plt.legend()
plt.show()


# Plot test error vs. dimension
plt.plot(dimensions, test_errors)
plt.title('Test Error vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Classification Error')
plt.show()

# Plot test error vs. dimension
plt.plot(dimensions, train_errors)
plt.title('Training Error vs. Dimension')
plt.xlabel('Dimension')
plt.ylabel('Classification Error')
plt.show()
