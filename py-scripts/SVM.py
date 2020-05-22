import pickle
import sys
import numpy as np
from nltk import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from collections import defaultdict
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
np.random.seed(500)

string = sys.argv[1]
string = string.lower()
string = word_tokenize(string)
word_Lemmatized = WordNetLemmatizer()

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV
Final_words = []

for word, tag in pos_tag(string):
    if word not in stopwords.words('english') and word.isalpha():
        word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
        Final_words.append(word_Final)

transformer = TfidfTransformer()
corpus = [" ".join(i for i in Final_words)]

vocabulary = pickle.load(open("py-scripts/models/feature.pkl", 'rb'))
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary, ngram_range=(1,3))),('tfid', TfidfTransformer())]).fit(corpus)
pipe['count'].transform(corpus).toarray()
final = pipe.transform(corpus)

with open("py-scripts/models/Pickle_SVC_Model.pkl", 'rb') as file:
    Pickled_SVC_Model = pickle.load(file)

prediction = Pickled_SVC_Model.predict(final)[0]
if prediction == 1:
    print("success")
else:
    print("danger")