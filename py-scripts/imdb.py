from tensorflow.keras.models import load_model
from keras.preprocessing import sequence
import sys
import json

text = sys.argv[1]

max_words = 10240
maxlen= 32

word_index_json = open("py-scripts/model-dependencies/imdb/imdb_word_index.json", "r")
word_index = json.load(word_index_json)

def get_sentiment(text):
    
    text = text.split(' ')
    
    indices = []
    for word in text:
        
        if word.lower() in word_index:
            
            indices.append(word_index[word.lower()])

    model = load_model('py-scripts/model-dependencies/imdb/imdb.h5')
    return (model.predict(sequence.pad_sequences([indices], maxlen=maxlen))[0])

sentiment = get_sentiment(text)

if sentiment > 0.5:
    print('danger')
else:
    print('success')