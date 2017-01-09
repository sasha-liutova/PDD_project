import mysql.connector
import pickle


def execute_query(query_text):
    cnx = mysql.connector.connect(user='so', password='mnanenatomamavychovalaabysomsedelvbaseodveceradorana',
                                  host='52.214.158.93', database='so', use_unicode=False)
    cursor = cnx.cursor()
    query = (query_text)
    cursor.execute(query)
    posts = []
    for post in cursor:
        posts.append(post[0].decode("utf-8") )
    cursor.close()
    cnx.close()
    return posts


def extract_code(posts):
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


def extract_features(snippets):
    pass


def save_data(data, filename):
    pickle.dump(data, open(filename, "wb"))


if __name__ == "__main__":

    posts = execute_query("SELECT Body FROM js_posts")
    raw_snippets = extract_code(posts)
    meaningful_snippets = filter_meaningful_snippets(raw_snippets)
    snippets = filter_js(meaningful_snippets)
    features = extract_features(snippets)
    save_data(snippets, 'snippets.py')
    save_data(features, 'features_?.py')


    # print('# snippets: ', len(meaningful_snippets))
    # for snippet in meaningful_snippets:
    #     print('------------------------')
    #     print(snippet)
    #
    # with open('snippets.txt', mode='w', encoding='utf-8') as a_file:
    #     for snippet in meaningful_snippets:
    #         a_file.write(snippet)