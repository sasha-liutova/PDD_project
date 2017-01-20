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
    snippets, snippets_trash, i = [], [], 0
    time0 = time()
    for post in cursor:
        snippets_current = extract_code(post[0])
        for snippet in snippets_current:
            if is_meaningful(snippet):
                # snippets.append({'snippet': snippet, 'features': [post[1], post[2]]})
                snippets.append(snippet)
            else:
                snippets_trash.append(snippet)
        i += 1
        if i % 10000 == 0:
            print(i, ' posts processed, ', (time()-time0), ' s')
            time0 = time()
    print('Number of posts retrieved: ', i, ', number of snippets: ', len(snippets), ', filtered out snippets: ', len(snippets_trash))
    save_data(snippets_trash, 'data/snippets_trash.dat')
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
            snippets.append(post[start : end].replace('&#xA;', '\n').replace('&gt;', '>').replace('&lt;', '<'))
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
    with open('data/vocabulary_Bag_of_Words_mindf1_Full.txt', mode='w', encoding='utf-8') as a_file:
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
    with open('data/vocabulary_Tf_idf_mindf2_Full.txt', mode='w', encoding='utf-8') as a_file:
            a_file.write(vocabulary)
    return features


def save_data(data, filename):
    pickle.dump(data, open(filename, "wb"))


def substitute_symbols(snippets):
    new_snippets = []
    for snippet in snippets:
        new_snippet = snippet.replace('&gt;', '>').replace('&lt;', '<')
        new_snippets.append(new_snippet)
    return new_snippets


def filter_out_xml(snippets):
    new_snippets, filtered_out = [], []
    for snippet in snippets:
        if snippet[0] in ['<', '>'] or snippet[-1] in ['<', '>']:
            filtered_out.append(snippet)
        else:
            new_snippets.append(snippet)
    print('# filtered out xml: ', len(filtered_out), ', left: ', len(new_snippets))
    pickle.dump(filtered_out, open('data/filtered_out_xml_snippets.dat', "wb"))
    return new_snippets


if __name__ == "__main__":

    t0 = time()
    print('Retrieving data from database...')
    snippets_raw = load_snippets("SELECT Body FROM js_posts_full")
    print(time() - t0, ' s')
    # snippets = load_snippets("SELECT Body, Score, ViewCount, AnswerCount, CommentCount FROM js_posts_full")
    # print('Saving snippets...')
    # save_data(snippets, 'snippets_extended.dat')
    # snippets = pickle.load(open('data/snippets.dat', "rb"))

    print('Filtering out XML snippets...')
    t0 = time()
    snippets = filter_out_xml(snippets_raw)
    print('After filtering out XML ', len(snippets), ' out of ', len(snippets_raw), ' left.')
    print(time() - t0, ' s')

    print('Extracting tf_idf features from snippets...')
    t0 = time()
    features1 = extract_features_tf_idf(snippets)
    print(time() - t0, ' s')

    # print('Saving features...')
    # save_data(features1, 'data/features_tf_idf.dat')

    print('Extracting bag_of_words features from snippets...')
    t0 = time()
    features2 = extract_features_bag_of_words(snippets)
    print(time() - t0, ' s')

    # print('Saving features...')
    # save_data(features2, 'data/features_bag_of_words.dat')
