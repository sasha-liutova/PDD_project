from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import numpy as np
import pickle


def load_data(filename):
    """
    Loads data from file
    """
    data = pickle.load(open(filename, "rb"))
    return data


def cluster_k_means(data, n_clusters):
    """
    Performs K-Means clustering on data
    :return: list of labels
    """
    kmeans = KMeans(n_clusters=n_clusters)
    labels = kmeans.fit_predict(data)
    return labels


def cluster_DBSCAN(data):
    """
    Performs DBSCAN clustering on data
    :return: list of labels
    """
    dbscan = DBSCAN()
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
    transformed_data = model.fit_transform(data.toarray())
    return transformed_data


def visualise(data, labels):
    """
    Plots a graph, where each point represents a snippet from data and
    is positioned in 2D space and colored according to its cluster.
    """

    # calculate how many clusters there are
    unique_labels = []
    for l in labels:
        if not l in unique_labels:
            unique_labels.append(l)
    n_clusters = len(unique_labels)
    print('# of clusters: ', n_clusters)

    # normalizing clusters' numbers fo colors assigning
    minimum = min(unique_labels)
    labels_normalized = [x - minimum for x in labels]

    # creating structure for displaying, containing pairs of coordinates and cluster numbers
    plot_data = []
    for label, point in zip(labels_normalized, data):
        plot_data.append([point, label])

    # list of colors for visualization
    colors = ['#a6cee3', '#1f78b4', '#b2df8a', '#33a02c', '#fb9a99', '#e31a1c',
              '#fdbf6f', '#ff7f00', '#cab2d6', '#6a3d9a', '#ffff99', '#b15928']
    if n_clusters > len(colors):
        color = iter(plt.cm.rainbow(np.linspace(0, 1, n_clusters)))
        colors = []
        for i in range(n_clusters):
            c = next(color)
            colors.append(c)

    # displaying plot
    for element in plot_data:
        point, label = element[0], element[1]
        plt.scatter(point[0], point[1], color=colors[label], s=5)
    plt.show()



if __name__ == "__main__":
    n_clusters = 10
    print('Loading data from files...')
    features, snippets = load_data('features_bag_of_words.dat'), load_data('snippets.dat')
    print('Performing clustering...')
    labels = cluster_k_means(features, n_clusters)
    print('Saving clustered data...')
    save_clustered_snippets(snippets, labels, n_clusters, 'clusters_KMeans_1.txt')
    print('Transforming data to 2D space...')
    transformed_features = reduce_dimensionality(features)
    print('Visualizing data...')
    visualise(transformed_features, labels)