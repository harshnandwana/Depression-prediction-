
import nltk
from nltk.corpus import stopwords 
from nltk.stem import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import string 
import re
import pickle
import joblib
import numpy as np
import streamlit as stl
stl.title("depression predictor")

def cleanhtml(phrase):
    text=re.compile('<.*?>')
    phrase=re.sub(text,' ',phrase)
    return phrase

def decont(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)
    phrase = re.sub(r"PRON", " ", phrase)
    phrase = re.sub(r"pron", " ", phrase)
    #remove any url
    phrase = re.sub(r"http[s]?://\S+"," ", phrase)
    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)

    #special char
    phrase = re.sub(r'[^A-Za-z0-9]+', " ", phrase)

    phrase = re.sub(r"@+"," ", phrase)

    #remove any thing with html tags
    phrase = cleanhtml(phrase)
    return phrase
data_load_state = stl.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
lr = joblib.load('model.pkl')
w2v_model = joblib.load('vectors.pkl')
w2v_words = joblib.load('words.pkl')
nltk.download('stopwords')
nltk.download('punkt')
import spacy
nlp = spacy.load('en_core_web_sm')
lmtzr = WordNetLemmatizer()

sno= nltk.stem.SnowballStemmer('english')
data_load_state.text('Loading data...done!')
def text_preprocess(text):
    lm = []
    text = nlp(text)
    for token in text:
        k=(token.lemma_)
        k=decont(k)
        #k=sno.stem(k)
        lm.append(k)
    text = " ".join(lm)
    text = text.translate(str.maketrans("", "", string.punctuation))
    #text = [word for word in text.split() ]#if word.lower() not in stopwords.words('english')]
    text = [word for word in text.split() if word.lower() not in stopwords.words('english')]
    #print(text)
    return " ".join(text)



def predict(a):
  a=text_preprocess(a)
  print(a)
  avg_w2v = []; # the avg-w2v for each sentence/review is stored in this list
  vector = np.zeros(50) # as word vectors are of zero length
  cnt_words =0; # num of words with a valid vector in the sentence/review
  for word in a.split(" "): # for each word in a review/sentence
      if word in w2v_words:
          vector += w2v_model[word]
          cnt_words += 1
  if cnt_words != 0:
      vector /= cnt_words
  avg_w2v.append(vector)

  val=lr.predict(avg_w2v)
  if val[0]==1:
    prediction = "Not depressed"
  else:
    prediction = "Depressed"
  return prediction

a=stl.text_input("enter text here") 
stl.subheader(predict(a))
