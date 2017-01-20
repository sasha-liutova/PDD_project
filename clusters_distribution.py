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

    # labels = pickle.load(open('data/labels_BoW_40.dat', "rb"))
    # snippets = pickle.load(open('data/snippets.dat', "rb"))
    # print(Counter(labels))
    # clusters = [38, 35, 34, 31, 30, 28, 27, 22, 21, 16, 15, 14, 13, 12, 8, 7, 5, 4, 3, 2, 1]
    # text=''
    # for label in clusters:
    #     index = numpy.where(labels == label)[0][0]
    #     snippet = snippets[index]
    #     text += snippet + '\n' + separator
    # n_lines = text.count('\n')
    # print('# lines: ', n_lines, ', # snippets: ', len(clusters), ', # lines per snippet: ', n_lines/len(clusters))
    # with open('data/single_snippet_clusters.txt', mode='w', encoding='utf-8') as a_file:
    #      a_file.write(text)


    # labels = pickle.load(open('data/labels_tf_idf_40.dat', "rb"))
    # snippets = pickle.load(open('data/snippets.dat', "rb"))
    # print(Counter(labels))
    # label = 38
    # text = ''
    # indexes = numpy.where(labels == label)[0]
    # for index in indexes:
    #     snippet = snippets[index]
    #     text += snippet + '\n' + separator
    # n_lines = text.count('\n')
    # with open('data/content_analysis_tfidf_40.txt', mode='w', encoding='utf-8') as a_file:
    #     a_file.write(text)