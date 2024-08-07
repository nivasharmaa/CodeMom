import random
import json
import pickle
import numpy as np
import tensorflow as tf
import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))  #load all the words
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot.h5')

# a function for cleaning up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words  
  

# a function to create the bag of words: it converts a sentence into a bag of words, a list full of zeroes and ones that indicates if the word is there or not
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# a function for predicting the class based on the bag of words
def predict_class(sentence): #from the bag of words, predicts, does not have too much certainty to provide better results in terms of probability
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# a function for getting the response based on the predicted class
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("No, we don't make 'yo mama' jokes here(maybe). Ask us anything!")
while True:
    message = input("")
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    