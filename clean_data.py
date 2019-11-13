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