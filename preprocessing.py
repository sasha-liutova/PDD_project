import mysql.connector
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time


def load_snippets(query_text):
    """
    Retrieves posts from database and extracts meaningful snippets
    :param query_text:
    :return: a list of snippets
    """
    cnx = mysql.connector.connect(user='so', password='mnanenatomamavychovalaabysomsedelvbaseodveceradorana',
                                  host='52.214.158.93', database='so')
    cursor = cnx.cursor()
    cursor.execute(query_text)
    snippets, i = [], 0
    time0 = time()
    for post in cursor:
        snippets_current = extract_code(post[0])
        for snippet in snippets_current:
            if is_meaningful(snippet):
                snippets.append(snippet)
        i += 1
        if i % 10000 == 0:
            print(i, ' posts processed, ', (time()-time0)/(i/10000), ' s')
        # if i % 130000 == 0:
        #     save_data(snippets, 'snippets' + str(i/130000) + '.dat')
        #     print('snippets' + str(i/130000) + '.dat saved')
        #     snippets = []

    print('Number of posts retrieved: ', i, ', number of snippets: ', len(snippets))
    cursor.close()
    cnx.close()
    return snippets


def extract_code(post):
    """
    Extracts snippets from post.
    """
    start_str, end_str = '<code>', '</code>'
    snippets = []
    start = post.find(start_str)
    while start >= 0:
        start += len(start_str)
        end = post.find(end_str, start)
        if end > 0:
            snippets.append(post[start : end].replace('&#xA;', '\n'))
        start = post.find(start_str, end)
    return snippets


def is_meaningful(snippet):
    if len(snippet) > 20:
            return True
    return False


def is_letter(character):
    """
    returns True if character is a letter or underscore
    """
    c = ord(character)
    if (64 < c < 91) or (96 < c < 123) or c == 95:
        return True
    return False


def extract_words(snippet):
    word, words = '',''
    for character in snippet:
        if is_letter(character):
            word += character
        else:
            if word != '':
                words += word + ' '
                word = ''
    return words


def extract_features_bag_of_words(snippets):
    """
    Performs feature extraction using Bag of Words algorithm.
    :param snippets:
    :return: features
    """
    vectorizer = CountVectorizer(min_df=2) # a word has to occur in at least 2 documents to be added to vocabulary
    features = vectorizer.fit_transform(snippets)
    vocabulary = str(vectorizer.vocabulary_)
    with open('vocabulary_Bag_of_Words_mindf1_Full.txt', mode='w', encoding='utf-8') as a_file:
            a_file.write(vocabulary)
    return features


def extract_features_tf_idf(snippets):
    """
    Performs feature extraction using Tf-idf algorithm.
    :param snippets:
    :return: features
    """
    vectorizer = TfidfVectorizer(min_df=2)
    features = vectorizer.fit_transform(snippets)
    vocabulary = str(vectorizer.vocabulary_)
    with open('vocabulary_Tf_idf_mindf2_Full.txt', mode='w', encoding='utf-8') as a_file:
            a_file.write(vocabulary)
    return features


def save_data(data, filename):
    pickle.dump(data, open(filename, "wb"))


if __name__ == "__main__":

    print('Retrieving data from database...')
    snippets = load_snippets("SELECT Body FROM js_posts_full")
    print('Saving snippets...')
    save_data(snippets, 'snippets.dat')
    print('Extracting features from snippets...')
    features = extract_features_bag_of_words(snippets)
    print('Saving features...')
    save_data(features, 'features_bag_of_words.dat')
