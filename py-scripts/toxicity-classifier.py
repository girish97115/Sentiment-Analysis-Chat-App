import sys
from keras.models import load_model
import h5py
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

text = sys.argv[1] 

train = pd.read_csv('py-scripts/model-dependencies/toxicity-classifier/train.csv')
tokenizer = Tokenizer(num_words=None,
                      filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                      lower=True,
                      split=" ",
                      char_level=False)
tokenizer.fit_on_texts(list(train['comment_text']))

model = load_model("py-scripts/model-dependencies/toxicity-classifier/toxicity_classifier.h5")
model.load_weights('py-scripts/model-dependencies/toxicity-classifier\cp.ckpt')

def toxicity_level(string):
    """
    Return toxicity probability based on inputed string.
    """
    #Process string
    new_string = [string]
    new_string = tokenizer.texts_to_sequences(new_string)
    new_string = pad_sequences(new_string, maxlen=380, padding='post', truncating='post')

    prediction = model.predict(new_string)
    toxic = False
    for toxicity_category in prediction[0]:

        if toxicity_category > 0.20:
            toxic = True
    
    return toxic

toxicity = toxicity_level(text)
if toxicity:
    print("danger")
else:
    print("success")