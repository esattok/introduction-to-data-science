import pandas as pd
import numpy as np
import sklearn.decomposition
import sklearn.cluster
import matplotlib.pyplot as plt

# Loading data
df = pd.read_csv('falldetection_dataset.csv', header=None)
df = df.iloc[:, 1:]

# Preprocessing labels
labels = np.array(df.loc[:,1]).reshape(566,1) 
labels = np.where(df.loc[:, 1] == 'F', 1, 0).reshape(-1, 1)
# Preprocessing features
features = df.iloc[:, 2:].values

# Perform PCA on full data
pca = sklearn.decomposition.PCA(n_components=2)
variance_full = pca.fit_transform(features)

# Plot 2-D Plot with Initial Labels
fails = labels[:, 0] == 1
nofails = labels[:, 0] == 0

plt.style.use('seaborn-pastel')
plt.scatter(variance_full[fails, 0], variance_full[fails, 1], label='F',   c='#EE7AE9')
plt.scatter(variance_full[nofails, 0], variance_full[nofails, 1], label='NF',  c='#7CCD7C')
plt.title("Initial Labels")
plt.legend()
plt.show()

# Remove outliers
outlier_index1 = np.argmax(variance_full[:, 0])
outlier_index2 = np.argmax(variance_full[:, 1])

new_features = np.delete(variance_full, [outlier_index1, outlier_index2], axis=0)
new_labels = np.delete(labels, [outlier_index1, outlier_index2]).reshape(len(new_features), 1)

# Plot 2-D Plot with Initial Labels and without Outliers
fails_new = new_labels[:, 0] == 1
nofails_new = new_labels[:, 0] == 0

plt.scatter(new_features[fails_new, 0], new_features[fails_new, 1], label='F', c='#EE7AE9')
plt.scatter(new_features[nofails_new, 0], new_features[nofails_new, 1], label='NF', c='#7CCD7C')
plt.title("Initial Labels & w/o Outliers")
plt.legend()
plt.show()

# Perform PCA on non-outlier data
variance_nooutlier = pca.fit_transform(new_features)

# Plot Dimension Reduction Without Outliers
fails_nooutlier = new_labels[:, 0] == 1
nofails_nooutlier = new_labels[:, 0] == 0

plt.scatter(variance_nooutlier[fails_nooutlier, 0], variance_nooutlier[fails_nooutlier, 1], label='F', c='#EE7AE9')
plt.scatter(variance_nooutlier[nofails_nooutlier, 0], variance_nooutlier[nofails_nooutlier, 1], label='NF', c='#7CCD7C')
plt.title("Dimension Reduction w/o Outliers")
plt.legend()
plt.show()

# K-means clustering on full data
clusters = 2
kmeans = sklearn.cluster.KMeans(n_clusters=clusters)
kmeans.fit(variance_full)
pred_lab = kmeans.labels_

# Plot Full Data KMeans Clustering n=2
fails_kmeans_full = pred_lab == 1
nofails_kmeans_full = pred_lab == 0

plt.scatter(variance_full[fails_kmeans_full, 0], variance_full[fails_kmeans_full, 1], label="F", c='#EE7AE9')
plt.scatter(variance_full[nofails_kmeans_full, 0], variance_full[nofails_kmeans_full, 1], label="NF", c='#7CCD7C')
plt.title("K-means Clustring for n=2 with All Data")
plt.legend()
plt.show()

# Calculate accuracy of K-means for full data
accuracy_full = np.sum(pred_lab != labels.reshape(-1)) / len(labels)
print("Accuracy of K-means For Full Data:", accuracy_full)

# K-means clustering on non-outlier data
kmeans.fit(variance_nooutlier)
pred_lab_nooutlier = kmeans.labels_

# Plot Non-outlier Data KMeans Clustering n=2
fails_kmeans_nooutlier = pred_lab_nooutlier == 0
nofails_kmeans_nooutlier = pred_lab_nooutlier == 1

plt.scatter(variance_nooutlier[fails_kmeans_nooutlier, 0], variance_nooutlier[fails_kmeans_nooutlier, 1], label="F", c='#EE7AE9')
plt.scatter(variance_nooutlier[nofails_kmeans_nooutlier, 0], variance_nooutlier[nofails_kmeans_nooutlier, 1], label="NF", c='#7CCD7C')
plt.title("K-means Clustring for n=2 with Non-outlier Data ")
plt.legend()
plt.show()

# Calculate accuracy of K-means for non-outlier data
accuracy_nooutlier = np.sum(pred_lab_nooutlier == new_labels.reshape(-1)) / len(new_labels)
print("Accuracy of K-means For Non-Outlier Data:", accuracy_nooutlier)

# K-means clustering for k = {4, 6, 8, 10, 12}
clusters = np.array([4, 6, 8, 10, 12], dtype=int)

for k in clusters:
    kmeans = sklearn.cluster.KMeans(n_clusters=k)
    kmeans.fit(variance_full)
    pred_lab = kmeans.labels_

    # Plot K-means Results for k = {4, 6, 8, 10, 12}
    labels_kmeans = np.unique(pred_lab)
    filename = f"K-means Clustring for n={k} with All Data.png"

    for i in labels_kmeans:
        plt.scatter(variance_full[pred_lab == i, 0], variance_full[pred_lab == i, 1], label=str(i))

    plt.title(filename)
    plt.legend()
    plt.show()

