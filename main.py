import mysql.connector

def extract_code(text):
    start_str, end_str = '<code>', '</code>'
    code = []
    start = text.find(start_str)
    while start >= 0:
        start += len(start_str)
        end = text.find(end_str, start)
        if end > 0:
            code.append(text[start : end])
        start = text.find(start_str, end)
    return code

cnx = mysql.connector.connect(user='so', password='mnanenatomamavychovalaabysomsedelvbaseodveceradorana',
                              host='52.210.148.46', database='so')
cursor = cnx.cursor()
query = ("SELECT Body FROM js_posts")
cursor.execute(query)
posts = []
for post in cursor:
  posts.append(post[0])
cursor.close()
cnx.close()

snippets = []
for post in posts:
    extracted_snippets = extract_code(post)
    # print(len(extracted_snippets), ': ', extracted_snippets)
    snippets.extend(extracted_snippets)

print('# snippets: ', len(snippets))
# print(snippets[7])
for snippet in snippets:
    print('------------------------')
    print(snippet)

# cursor = cnx.cursor()
# query = ("SELECT Body FROM js_posts WHERE Id = 845")
# cursor.execute(query)
# posts = []
# for post in cursor:
#   posts.append(post)
# print(posts)
# cursor.close()
# cnx.close()

