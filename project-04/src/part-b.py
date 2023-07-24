import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('falldetection_dataset.csv', header=None)
df = df.iloc[:, 1:]

labels = df.iloc[:, 0].values
features = df.iloc[:, 1:].values

no_falls_indices = np.argwhere(labels == 'NF').reshape(-1)
fall_indices = np.argwhere(labels == 'F').reshape(-1)

train_nf, test_nf = train_test_split(no_falls_indices, test_size=38)
train_f, test_f = train_test_split(fall_indices, test_size=47)

train_indices = np.concatenate((train_nf, train_f))
test_indices = np.concatenate((test_nf, test_f))

train_features = features[train_indices]
train_labels = labels[train_indices]
test_features = features[test_indices]
test_labels = labels[test_indices]

validation_size = 47
validation_features, test_features, validation_labels, test_labels = train_test_split(test_features, test_labels, test_size=validation_size)

svm = SVC(C=1.0, degree=12, max_iter=10000, shrinking=False, kernel = 'linear')
svm.fit(train_features, train_labels)

validation_accuracy = svm.score(validation_features, validation_labels)
print("Hyperparameters: C = 1.0, degree = 12, max_iter = 10000, shrinking = False, kernel = 'linear'")
print("SVM Validation Accuracy: {:.5f}%".format(validation_accuracy * 100))

test_accuracy = svm.score(test_features, test_labels)
print("SVM Test Accuracy: {:.5f}%".format(test_accuracy * 100))
#*******************************
mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, learning_rate_init=0.1, max_iter=200)
mlp.fit(train_features, train_labels)

validation_accuracy = mlp.score(validation_features, validation_labels)
print("Hyperparameters: solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=42, learning_rate_init = 0.1, max_iter = 200")
print("MLP Validation Accuracy: {:.5f}%".format(validation_accuracy * 100))

test_accuracy = mlp.score(test_features, test_labels)
print("MLP Test Accuracy: {:.5f}%".format(test_accuracy * 100))
