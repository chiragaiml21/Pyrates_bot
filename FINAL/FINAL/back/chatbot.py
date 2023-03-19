import random 
import json 
import pickle
import numpy as np 
from tensorflow import keras
import streamlit as st

import nltk 
from nltk.stem import WordNetLemmatizer

from tensorflow import keras
from keras.models import load_model


lemmatizer=WordNetLemmatizer()
with open('back/intents.json', encoding='utf8') as f:
    intents = json.load(f)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbot_model.model')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag=[0]*len(words)
    for w in sentence_words:
        for i , word in enumerate(words):
            if word ==w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow=bag_of_words(sentence)
    
    res = model.predict(np.array([bow]))[0]
    error_threshold=0.25
    results=[[i,r] for i,r in enumerate(res) if r > error_threshold]

    results.sort(key=lambda x: x[1],reverse= True)
    return_list = []
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability':str(r[1])})

    tag=return_list[0]['intent']
    list_of_intents=intents['intents']
    flag=0
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            flag=1
            break
    if flag==0:
        print("i do not understand")
        # else:
        #     result = "Sorry, I don't have the answer of you query yet, but I'll work on it to make your experience better in future."
    return result

    
# print('welcome')


