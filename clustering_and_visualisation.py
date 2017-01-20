from sklearn.cluster import KMeans, DBSCAN, MiniBatchKMeans
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import pickle
from sklearn.decomposition import PCA, TruncatedSVD
import time


def load_data(filename, n_snippets):
    """
    Loads specified number of snippets from file
    """
    data = pickle.load(open(filename, "rb"))
    data_reduced = data[: n_snippets]
    return data_reduced


def cluster_k_means(data, n_clusters):
    """
    Performs K-Means clustering on data
    :return: list of labels
    """
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels


def cluster_minibatchkmeans(data, n_clusters):
    """
    Performs MiniBatchKMeans clustering on data
    :return: list of labels
    """
    km = MiniBatchKMeans(n_clusters=n_clusters, init='k-means++', n_init=3,
                         init_size=1000, batch_size=1000)
    labels = km.fit_predict(data)
    return labels


def cluster_DBSCAN(data, eps):
    """
    Performs DBSCAN clustering on data
    :return: list of labels
    """
    dbscan = DBSCAN(eps=eps, min_samples=5)
    labels = dbscan.fit_predict(data)
    return labels


def save_clustered_snippets(snippets, labels, n_clusters, filename):
    """
    Saves clustered snippets to file for possible further analysis.
    Snippets are ordered by their cluster.
    :param snippets:
    :param labels:
    :param n_clusters: number of clusters
    :param filename: destination file name
    :return:
    """

    if n_clusters is None:
        # calculate how many clusters there are
        unique_labels = []
        for l in labels:
            if not l in unique_labels:
                unique_labels.append(l)
        n_clusters = len(unique_labels)

    separator = ''
    for i in range(50):
        separator += '-'
    separator += '\n'

    clustered_snippets = []
    for n in range(n_clusters):
        start_str = '\n\n' + separator + 'CLUSTER # ' + str(n) + '\n' + separator
        clustered_snippets.append(start_str)


    for snippet, label in zip(snippets, labels):
        clustered_snippets[label] += snippet + '\n' + separator


    with open(filename, mode='w', encoding='utf-8') as a_file:
        for cluster in clustered_snippets:
                a_file.write(cluster)


def reduce_dimensionality(data):
    """
    Reduces dimensionality of data to 2D for visualisation purposes
    """
    model = TSNE(n_components=2)
    transformed_data = model.fit_transform(data)
    return transformed_data


def visualise(data, labels):
    """
    Plots a graph, where each point represents a snippet from data and
    is positioned in 2D space and colored according to its cluster.
    """

    # calculate how many clusters there are
    unique_labels = []
    counters = {}
    for l in labels:
        if not l in unique_labels:
            unique_labels.append(l)
            counters[l] = 0
        else:
            counters[l] += 1
    n_clusters = len(unique_labels)
    print('# of clusters: ', n_clusters)
    unique_labels = sorted(counters, key=counters.get, reverse=True)

    # normalizing clusters' numbers fo colors assigning
    minimum = min(unique_labels)
    labels_normalized = [x - minimum for x in labels]

    # creating structure for displaying, containing pairs of coordinates and cluster numbers
    plot_data = []
    for label, point in zip(labels_normalized, data):
        plot_data.append([point, label])

    # list of colors for visualization
    colors = ['#a6cee3', '#33a02c', '#fdbf6f', '#b2df8a', '#fb9a99', '#cab2d6', '#ffff99',
              '#1f78b4', '#e31a1c', '#ff7f00', '#6a3d9a', '#b15928']
    if n_clusters > len(colors):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, n_clusters)))
        for i in range(n_clusters - len(colors)):
            c = next(color)
            colors.append(c)

    # displaying plot
    for element in plot_data:
        point, label = element[0], element[1]
        index = unique_labels.index(label)
        plt.scatter(point[0], point[1], color=colors[index], s=3)
    plt.show()


def apply_TruncatedSVD(data):
    svd = TruncatedSVD(n_components=10, n_iter=7, random_state=42)
    return svd.fit_transform(data)
    # print(svd.explained_variance_ratio_)
    # print(svd.explained_variance_ratio_.sum())



if __name__ == "__main__":
    n_clusters, n_snippets = 40, 20000

    print('Loading data from files...')
    t0 = time.time()
    features = load_data('data/features_tf_idf.dat', n_snippets)
    snippets = load_data('data/snippets.dat', n_snippets)
    print(time.time() - t0, ' s')

    print('Performing clustering...')
    t0 = time.time()
    labels = cluster_k_means(features, n_clusters)
    pickle.dump(labels, open('data/labels_tf_idf_' + str(n_clusters) + '.dat', "wb"))
    print(time.time() - t0, ' s')
    # labels = pickle.load(open('data/labels_tf_idf_' + str(n_clusters) + '.dat', "rb"))

    print('Saving clustered data...')
    t0 = time.time()
    save_clustered_snippets(snippets, labels, n_clusters, 'data/clusters_tf_idf_' + str(n_clusters) + '.txt')
    print(time.time() - t0, ' s')

    print('Applying Truncated SVD...')
    t0 = time.time()
    reduced_features = apply_TruncatedSVD(features)
    print(time.time() - t0, ' s')

    print('Transforming data to 2D space...')
    t0 = time.time()
    transformed_features = reduce_dimensionality(reduced_features)
    pickle.dump(transformed_features, open('data/data_vizual_tf_idf.dat', "wb"))
    print(time.time() - t0, ' s')
    # transformed_features = pickle.load(open('data/data_vizual_tf_idf.dat', "rb"))

    print('Visualizing data...')
    visualise(transformed_features, labels)