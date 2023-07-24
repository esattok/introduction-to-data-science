import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np


def read_mnist_images(filename):
    return np.loadtxt(filename).reshape(-1, 400)


def read_mnist_labels(filename):
    return np.loadtxt(filename)


def plot_tsne(digits_data, labels_data, perplexity, n_iter, title):
    tsne = TSNE(n_components=2, random_state=64, perplexity=perplexity, n_iter=n_iter)
    embedded_data = tsne.fit_transform(digits_data)

    plt.scatter(embedded_data[:, 0], embedded_data[:, 1], c=labels_data, cmap='jet')
    plt.title(title)
    plt.colorbar()
    plt.show()


# Example usage
digits_data = read_mnist_images('./digits/digits.txt')
labels_data = read_mnist_labels('./digits/labels.txt')

# Create scatter plots of the embedded data for different perplexity and n_iter values
perplexities = [50.0, 30.0, 50.0, 30.0, 50.0, 30.0]
n_iters = [1500, 1500, 3000, 3000, 4500, 4500]
titles = ['t-SNE Mapping for 1500 Iterations with 50 Perplexity',
          't-SNE Mapping for 1500 Iterations with 30 Perplexity',
          't-SNE Mapping for 3000 Iterations with 50 Perplexity',
          't-SNE Mapping for 3000 Iterations with 30 Perplexity',
          't-SNE Mapping for 4500 Iterations with 50 Perplexity',
          't-SNE Mapping for 4500 Iterations with 30 Perplexity']

for i in range(len(perplexities)):
    plot_tsne(digits_data, labels_data, perplexities[i], n_iters[i], titles[i])

print("done")
