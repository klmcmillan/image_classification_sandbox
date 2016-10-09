from __future__ import division
import os
import sys
import numpy as np
import pandas as pd
from random import randint
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Resize to average size of the images downloaded from Bing
STANDARD_SIZE = (230, 170)

def img_to_matrix(filename):
    """
    Takes an image name and turns image into a numpy array of RGB pixels.
    """
    img = Image.open(filename)
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)

    return img


def flatten_image(img):
    """
    Takes in an (m, n) numpy array and flattens it into an array of shape
    (1, m * n).
    """
    s = img.shape[0] * img.shape[1]
    img_wide = img.reshape(1, s)

    return img_wide[0]


def build_dataset(data_dir):
    """
    Organize the classification data in a format similar to sklearn datasets.
    Dataset is a dictionary consisting of the following:
    (1) target_names: array of unique label names
    (2) data: flattened array of RGB pixels for each image
    (3) target: integer representation of the label of each image
    """
    images = [data_dir+'/'+f for f in os.listdir(data_dir) if f != '.DS_Store']
    labels = [f.split('/')[-1].split('_')[0] for f in images]

    target_names = np.unique(np.array(labels))
    target = np.array(labels)
    target_num = 0
    for name in target_names:
        target[np.where(target==name)] = target_num
        target_num += 1
    target = target.astype(np.int)
    target = target.tolist() # necessary? just trying to match sklearn iris

    data = []
    for image in images:
        img = img_to_matrix(image)
        img = flatten_image(img)
        data.append(img)

    data = np.array(data)
    dataset = {'target_names': target_names, 'data': data, 'target': target}

    return dataset


def split_dataset(dataset, test_size):
    """
    Split the dataset into training and testing sets. Specify the fraction of
    testing data using the test_size variable (e.g. test_size = 0.4 for 40% test
    data and 60% training data).
    """
    X_train, X_test, y_train, y_test = model_selection.train_test_split(
        dataset['data'], dataset['target'], test_size=test_size, random_state=42)

    return X_train, X_test, y_train, y_test


def reduce_dim(X_train, X_test, n_components=5):
    """
    Use randomized PCA to reduce the dimmensions of the image data to a
    n_components set of dimmensions.
    """
    pca = PCA(n_components=n_components, svd_solver='randomized')
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def knn_classification(X_train, y_train, n_neighbors=15):
    """
    Create knn classifier using training data and n_neighbors number of neighbors.
    """
    knn = KNeighborsClassifier(n_neighbors=n_neighbors)
    knn.fit(X_train, y_train)

    return knn


def get_text_label(dataset, label_int):
    """
    Convert numercial labels from dataset back to text labels.
    """
    num_labels = len(dataset['target_names'])
    label_str = np.empty(np.shape(np.array(label_int)), dtype='S32')
    for i in np.arange(num_labels):
        label_str[np.where(label_int==i)] = dataset['target_names'][i]

    return label_str


def print_metrics(classifier, dataset, X_test, y_test):
    """
    Print the following:
    (1) Summary of precision, recall and F1 score for each class
    (2) Frequency table of predicted vs actual classifications for each class
    """
    y_actual = get_text_label(dataset, y_test)
    y_predicted = get_text_label(dataset, classifier.predict(X_test))

    print metrics.classification_report(y_actual, y_predicted)

    df = pd.DataFrame({"actual": y_actual, "predicted": y_predicted})
    print pd.crosstab(df.actual, df.predicted, margins=True)


def plot_classification(dataset, test_size=0.4, n_mesh=100, n_neighbors=15):
    """
    Use nearest neighbors classification to plot the decision boundaries
    for each class. Uses training data to generate classifications, so set
    test_size = 0 to use entire training set.
    """
    X_train, _, y_train, _ = split_dataset(dataset, test_size)

    pca = PCA(n_components=2, svd_solver='randomized')
    X = pca.fit_transform(X_train)
    y = y_train

    # Create color maps: red, green, blue, yellow, cyan, magenta
    colors = ['#FF0000', '#00FF00', '#0000FF', '#FFFF00', '#00FFFF', '#FF00FF']
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF', '#FFFFAA', '#AAFFFF', '#FFAAFF'], N=len(dataset['target_names']))

    clf = KNeighborsClassifier(n_neighbors=n_neighbors)
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Step size in the mesh set to have n_mesh elements in the largest dimmension
    h = int(max(x_max-x_min, y_max-y_min))/n_mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": get_text_label(dataset, y)})
    for label, color in zip(dataset['target_names'], colors):
        mask = df['label']==label
        plt.scatter(df[mask]['x'], df[mask]['y'], c=color, label=label)
    plt.legend()

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    plt.title('knn classification (k = ' + str(n_neighbors) + ')')

    plt.show()


def main():
    """
    User-specified parameters:
    (1) data_dir: directory where image data is saved
    (2) test_size: fraction of total data that should be categorized as testing data
    (3) n_components: PCA set of dimmensions
    (4) n_neighbors: KNeighborsClassifier number of neighbors
    (5) n_mesh: number of elements in the meshgrid

    Returns:
    (1) Prints to screen a series of classification metrics
    (2) Plot of classification decision boundaries for each image class
    """
    data_dir = 'images'
    test_size = 0.4
    n_components = 5
    n_neighbors = 15
    n_mesh = 150

    dataset = build_dataset(data_dir)
    X_train, X_test, y_train, y_test = split_dataset(dataset, test_size=test_size)
    X_train, X_test = reduce_dim(X_train, X_test, n_components=n_components)
    knn = knn_classification(X_train, y_train, n_neighbors=n_neighbors)

    print_metrics(knn, dataset, X_test, y_test)
    plot_classification(dataset, test_size=test_size, n_mesh=n_mesh, n_neighbors=n_neighbors)


if __name__ == '__main__':
    main()
    sys.exit(0)
