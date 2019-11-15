import sqlite3 as lite
con = lite.connect('./mxm_dataset.db')
cur = con.cursor()
QUERY = "SELECT * FROM lyrics" 
cur.execute(QUERY)


from sklearn.feature_extraction.text import TfidfVectorizer
i = 0
sentences = dict()
for track_id, mxm_tid, word, count, is_test in cur.fetchall():
    if is_test == 0:
        sentences[track_id] = sentences.get(track_id, "") + (word + " ")*count + " "
    if i > 1000:
        break
    i += 1

# print("sentences[:10]",sentences[:10])
sentences = list(sentences.values())
tf_idf = TfidfVectorizer().fit_transform(sentences)
print(tf_idf * tf_idf.T)
