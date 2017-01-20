import pickle
from collections import Counter
import matplotlib.pyplot as plt


def get_distribution(labels):
    distribution = list(Counter(labels).values())
    distribution.sort(reverse=True)
    return distribution


# def plot_distribution(data):
#
#     N = len(data)
#     x = range(N)
#     width = 1 / 1.5
#     plt.bar(x, data, width, color="blue")
#
#     plt.show()


if __name__ == "__main__":
    # labels = pickle.load(open('data/labels_tf_idf_10.dat', "rb"))
    # labels = pickle.load(open('data/labels_tf_idf_20.dat', "rb"))
    # labels = pickle.load(open('data/labels_tf_idf_30.dat', "rb"))
    # labels = pickle.load(open('data/labels_tf_idf_40.dat', "rb"))
    # labels = pickle.load(open('data/labels_BoW_10.dat', "rb"))
    # labels = pickle.load(open('data/labels_BoW_20.dat', "rb"))
    # labels = pickle.load(open('data/labels_BoW_30.dat', "rb"))
    labels = pickle.load(open('data/labels_BoW_40.dat', "rb"))

    distribution = get_distribution(labels)
    for a in distribution:
        print(a)