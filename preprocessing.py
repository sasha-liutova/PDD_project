import mysql.connector
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from time import time


def execute_query(query_text):
    """
    Executes a query specified in query_text.
    :param query_text:
    :return: a list of posts - result of query execution
    """
    cnx = mysql.connector.connect(user='so', password='mnanenatomamavychovalaabysomsedelvbaseodveceradorana',
                                  host='52.214.158.93', database='so')
    cursor = cnx.cursor()
    cursor.execute(query_text)
    print('Query executed. Postprocessing is performed...')
    posts, i = [], 0
    time0 = time()
    for post in cursor:
        posts.append(post[0])
        i += 1
        if i % 10000 == 0:
            print(i, ' posts processed, ', (time()-time0)/(i/10000), ' s')

    print('Number of posts retrieved: ', len(posts))
    cursor.close()
    cnx.close()
    return posts


def extract_code(posts):
    """
    Extracts snippets from posts.
    """
    start_str, end_str = '<code>', '</code>'
    snippets = []
    for text in posts:
        start = text.find(start_str)
        while start >= 0:
            start += len(start_str)
            end = text.find(end_str, start)
            if end > 0:
                snippets.append(text[start : end].replace('&#xA;', '\n'))
            start = text.find(start_str, end)
    return snippets


def filter_meaningful_snippets(snippets):
    filtered = []
    for snippet in snippets:
        if len(snippet) > 20:
            filtered.append(snippet)
    return filtered


def filter_js(snippets):
    return snippets


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
    posts = execute_query("SELECT Body FROM js_posts_full")
    print('Extracting snippets from posts...')
    raw_snippets = extract_code(posts)
    print('Extracting meaningful snippets...')
    meaningful_snippets = filter_meaningful_snippets(raw_snippets)
    print('Extracting JS snippets...')
    snippets = filter_js(meaningful_snippets)
    print('Saving snippets...')
    save_data(snippets, 'snippets.dat')
    print('Extracting features from snippets...')
    features = extract_features_bag_of_words(snippets)
    print('Saving features...')
    save_data(features, 'features_bag_of_words.dat')
