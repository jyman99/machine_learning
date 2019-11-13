import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sent = "This is a sample sentence, showing off the stop words filtration." # replace with data
import string
table = str.maketrans('', '', string.punctuation)
stripped = example_sent.translate(table)

# data already lowercase
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(stripped)
print(word_tokens)


filtered_sentence = [w for w in word_tokens if not w in stop_words]
print(filtered_sentence)

from spellchecker import SpellChecker
spell = SpellChecker()
filtered_sentence = [spell.correction(w)for w in filtered_sentence]
print(filtered_sentence)

from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

ps = PorterStemmer()

filtered_sentence = [ps.stem(w) for w in filtered_sentence]
print(filtered_sentence)

from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

filtered_sentence = [lemmatizer.lemmatize(w) for w in filtered_sentence]


print(filtered_sentence)





### maybe try adding PCA?
### make it work for our data

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def TFIDF(X_train, X_test, MAX_NB_WORDS=75000):
    vectorizer_x = TfidfVectorizer(max_features=MAX_NB_WORDS)
    X_train = vectorizer_x.fit_transform(X_train).toarray()
    X_test = vectorizer_x.transform(X_test).toarray()
    print("tf-idf with", str(np.array(X_train).shape[1]), "features")
    return (X_train, X_test)


from sklearn.datasets import fetch_20newsgroups

newsgroups_train = fetch_20newsgroups(subset='train') # replace with our data
newsgroups_test = fetch_20newsgroups(subset='test')   # replace with our data
X_train = newsgroups_train.data                       # replace with our data
X_test = newsgroups_test.data                         # replace with our data
y_train = newsgroups_train.target                     # replace with our data
y_test = newsgroups_test.target                       # replace with our data

X_train,X_test = TFIDF(X_train,X_test)

from sklearn.decomposition import PCA
pca = PCA(n_components=2000)
X_train_new = pca.fit_transform(X_train)
X_test_new = pca.transform(X_test)

print("train with old features: ",np.array(X_train).shape)
print("train with new features:" ,np.array(X_train_new).shape)

print("test with old features: ",np.array(X_test).shape)
print("test with new features:" ,np.array(X_test_new).shape)




