import sqlite3 as lite
import numpy as np
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel



# load track IDs and corresponding titles
mxm_titles = msd_titles = None
with open(".\mxm_titles.json") as f:
    mxm_titles = json.load(f)
with open(".\msd_titles.json") as f:
    msd_titles = json.load(f)



con = lite.connect('./mxm_dataset.db')
cur = con.cursor()

cur.execute("SELECT COUNT(*) FROM lyrics")
print("LYRICS COUNT:")
print(cur.fetchall()[0])



######################################################  SLOW AND UNNECESSARY  ############################
# title_con = lite.connect('./titles.db')
# title_cur = title_con.cursor()
# def has_title_and_artist_SQL(msd_track_id, mxm_track_id):
#     # TITLE_QUERY = "SELECT * FROM Titles"
#     TITLE_QUERY = 'SELECT 1 FROM Titles WHERE TrackId = "{}"'.format(msd_track_id)
#     title_cur.execute(TITLE_QUERY)
#     result = title_cur.fetchall()
#     # print('result1 =',result)
#     if len(result) > 0:
#         return result[0]
#     TITLE_QUERY = 'SELECT 1 FROM Titles WHERE MxmTrackId = "{}"'.format(mxm_track_id)
#     title_cur.execute(TITLE_QUERY)
#     result = title_cur.fetchall()
#     # print('result2 =',result)
#     if len(result) > 0:
#         return result[0]
#     print("NO SUCH ID FOUND")
#     return None
######################################################  SLOW AND UNNECESSARY  ############################



######################### EXAMPLE #################################
# from sklearn.datasets import fetch_20newsgroups
# twenty = fetch_20newsgroups()
# print()
# print('twenty.data.shape =',len(twenty.data), len(twenty.data[0]))
# print("\n")
# print('twenty =')
# print(twenty.data[0])
# print('-----')
# print(twenty.data[1])
# print('-----')
# tfidf = TfidfVectorizer().fit_transform(twenty.data)
# print('twenty tfidf shape =',tfidf.shape)
# twenty_cosine_similarities = linear_kernel(tfidf[0:1], tfidf).flatten()
# twenty_related_docs_indices = twenty_cosine_similarities.argsort()[:-5:-1]
# print(twenty_related_docs_indices)
# print(twenty_cosine_similarities[twenty_related_docs_indices])
######################### EXAMPLE #################################


def compute_similarities_text():

    QUERY = "SELECT * FROM lyrics LIMIT 50000" 
    cur.execute(QUERY)

    print("let's go!")
    i = 0
    id_to_sentence = dict()
    for msd_track_id, mxm_track_id, word, count, is_test in cur.fetchall():
        id_to_sentence[msd_track_id] = id_to_sentence.get(msd_track_id, "") + (word + " ")*count
            
    IDs, sentences = zip(*id_to_sentence.items())

    print()
    print('LYRICS METADATA')
    tf_idf = TfidfVectorizer().fit_transform(sentences)
    print('tf_idf.shape:',tf_idf.shape)

    print()
    print("TRACK TO COMPARE WITH:")
    print(id_to_sentence[IDs[0]])
    print(msd_titles[IDs[0]])
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()
    print("cosine_similarities.shape:", cosine_similarities.shape)
    print()

    ids_and_similarities = np.array(list(zip(list(cosine_similarities), IDs)))
    print('IDS AND SIMILARITIES:',ids_and_similarities[:3])
    print()

    sorted_ids_and_similarities = sorted(ids_and_similarities, key=lambda x:x[0])
    print(sorted_ids_and_similarities[0])
    print(sorted_ids_and_similarities[1])
    print(sorted_ids_and_similarities[-1])
    print(sorted_ids_and_similarities[-2])
    print(sorted_ids_and_similarities[-3])

    def print_top_n(n):
        i = 1
        for similarity_degree, track_id in sorted_ids_and_similarities[-n:][::-1]:
            try:
                title, artist = msd_titles[track_id]
                print()
                print("{}. {} by {} with similarity degree of {}".format(i, artist, title, similarity_degree))
                print(id_to_sentence[track_id])
                print()
                i += 1
            except Exception as e:
                print('error:',e)

    print_top_n(5)


def compute_similarities_text_remove_stopwords():

    QUERY = "SELECT * FROM lyrics LIMIT 5000" 
    cur.execute(QUERY)

    print("let's go!")
    i = 0
    id_to_sentence = dict()
    for msd_track_id, mxm_track_id, word, count, is_test in cur.fetchall():
        id_to_sentence[msd_track_id] = id_to_sentence.get(msd_track_id, "") + (word + " ")*count


    from nltk.corpus import stopwords
    # download('stopwords')
    stop_words=set(stopwords.words('english'))
    for k, v in id_to_sentence.items():
        id_to_sentence[k] = ' '.join([word for word in v.split() if word not in stop_words])
        id_to_sentence[k] = "Blank" if id_to_sentence[k] == "" else id_to_sentence[k]
    IDs, sentences = zip(*id_to_sentence.items())


    # sentences = [' '.join([word for word in sent.split() if word not in stop_words]) for sent in sentences]
    # print(sentences[:5])
    # print("IMPORTANTS")
    

    print()
    print('LYRICS METADATA')
    tf_idf = TfidfVectorizer().fit_transform(sentences)
    print('tf_idf.shape:',tf_idf.shape)

    print()
    print("TRACK TO COMPARE WITH:")
    print(id_to_sentence[IDs[0]])
    print(msd_titles[IDs[0]])
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()
    print("cosine_similarities.shape:", cosine_similarities.shape)
    print()

    ids_and_similarities = np.array(list(zip(list(cosine_similarities), IDs)))
    print('IDS AND SIMILARITIES:',ids_and_similarities[:3])
    print()

    sorted_ids_and_similarities = sorted(ids_and_similarities, key=lambda x:x[0])
    print(sorted_ids_and_similarities[0])
    print(sorted_ids_and_similarities[1])
    print(sorted_ids_and_similarities[-1])
    print(sorted_ids_and_similarities[-2])
    print(sorted_ids_and_similarities[-3])

    def print_top_n(n):
        i = 1
        for similarity_degree, track_id in sorted_ids_and_similarities[-n:][::-1]:
            try:
                title, artist = msd_titles[track_id]
                print()
                print("{}. {} by {} with similarity degree of {}".format(i, artist, title, similarity_degree))
                print(' '.join(sorted(id_to_sentence[track_id].split())))
                print()
                i += 1
            except Exception as e:
                print('error:',e)
        print('.')
        print('.')
        print('.')
        i = len(sorted_ids_and_similarities) - n
        for similarity_degree, track_id in sorted_ids_and_similarities[:n][::-1]:
            try:
                title, artist = msd_titles[track_id]
                print()
                print("{}. {} by {} with similarity degree of {}".format(i, artist, title, similarity_degree))
                print(' '.join(sorted(id_to_sentence[track_id].split())))
                print()
                i += 1
            except Exception as e:
                print('error:',e)

    print_top_n(5)

def compute_similarities_text_word2vec(train=False, learn_idx=None):

    QUERY = "SELECT * FROM lyrics LIMIT 5000" 
    cur.execute(QUERY)

    print("let's go!")
    i = 0
    id_to_sentence = dict()
    for msd_track_id, mxm_track_id, word, count, is_test in cur.fetchall():
        id_to_sentence[msd_track_id] = id_to_sentence.get(msd_track_id, "") + (word + " ")*count


    from nltk.corpus import stopwords
    # download('stopwords')
    stop_words=set(stopwords.words('english'))
    for k, v in id_to_sentence.items():
        id_to_sentence[k] = ' '.join([word for word in v.split() if word not in stop_words])
        id_to_sentence[k] = "Blank" if id_to_sentence[k] == "" else id_to_sentence[k]
    track_ids, sentences = zip(*id_to_sentence.items())

    from gensim.models import Word2Vec
    from gensim.models.doc2vec import Doc2Vec, TaggedDocument
    from gensim.test.utils import common_texts, get_tmpfile


    ############### Word2Vec #######################
    # path = get_tmpfile("word2vec.model")
    # model = Word2Vec(common_texts, size=100, window=5, min_count=1, workers=4)
    # model = Word2Vec.load("word2vec.model")
    # model.save("word2vec.model")



    ############### Doc2Vec ########################
    from nltk.tokenize import word_tokenize
    def learn_model(learn_idx):
        
        # from nltk import download
        # download("punkt")
        documents = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(sentences)]

        max_epochs = 10000
        vec_size = 300
        alpha = 0.040

        model = Doc2Vec(size=vec_size, alpha=alpha, min_alpha=0.00025, min_count=1, dm=0)
        # model = Doc2Vec(vector_size=vec_size, dm=0)
        model.build_vocab(documents)

        for epoch in range(max_epochs):
            print("iteration {0}".format(epoch))
            model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
            # model.alpha -= 0.0001
            # model.min_alpha = model.alpha
        
        model.save("doc2vec_bow{}.model".format(learn_idx))

    if train:
        if learn_idx != None:
            learn_model(learn_idx)
    else:
        model = Doc2Vec.load("doc2vec_bow{}.model".format(learn_idx))
        # test_doc = word_tokenize(sentences[0])
        # v1 = model.infer_vector(test_doc)
        # print("V1_infer:",v1)
        
        
        def print_top_n(x, n):
            title, artist = msd_titles[track_ids[x]]
            print('Examining --{} by {}--'.format(title, artist))
            print(' '.join(sorted(id_to_sentence[track_ids[x]].split())))
            print("------")
            test_doc = word_tokenize(id_to_sentence[track_ids[x]])
            v1 = model.infer_vector(test_doc)
            # print("V1_infer:",v1)
            similar_doc = model.docvecs.most_similar([v1], topn=n)
            print("len(similar_doc):",len(similar_doc))
            i = 1
            for idx, similarity_degree in similar_doc:
                idx = int(idx)
                title, artist = msd_titles[track_ids[idx]]
                print()
                print("{}. {} by {} with similarity degree of {}".format(i, title, artist, similarity_degree))
                print(' '.join(sorted(id_to_sentence[track_ids[idx]].split())))
                print()
                i += 1
                
        print_top_n(0, 5)

compute_similarities_text_word2vec(train=False, learn_idx=8)
