import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import json
with open('data.json', 'r') as file:
    data = json.load(file)
    
import string
import random 
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer 
import tensorflow as tf 
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Dropout
nltk.download("punkt")
nltk.download("wordnet")

lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_x = []
doc_y = []

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        tokens = nltk.word_tokenize(pattern)
        words.extend(tokens)
        doc_x.append(pattern)
        doc_y.append(intent["tag"])
    
    if intent["tag"] not in classes:
        classes.append(intent["tag"])

words = [lemmatizer.lemmatize(word.lower()) for word in words if word not in string.punctuation]

words = sorted(set(words))
classes = sorted(set(classes))

print(words)

['adios', 'anniversaire', 'as-tu', 'bye', "c'est", 'ce', 'cette', 'comment', 'cya', 'ensemble', 'es-tu', 'est', 'fais-tu', 'fasse', 'forme', 'hello', 'la', 'né', 'plan', 'pour', 'prénom', "qu'on", 'quand', 'que', 'quel', 'quels', 'qui', 'roule', 'salut', 'see', 'semaine', 'sont', "t'appelles", 'te', 'ton', 'truc', 'tu', 'un', 'veux', 'week-end', 'ya', 'yo', 'âge', 'ça']

print(classes)

['age', 'date', 'goodbye', 'greeting', 'name']

print(doc_x)

['Hello', 'La forme?', 'yo', 'Salut', 'ça roule?', 'Quel âge as-tu?', "C'est quand ton anniversaire?", 'Quand es-tu né?', 'Que fais-tu ce week-end?', "Tu veux qu'on fasse un truc ensemble?", 'Quels sont tes plans pour cette semaine', 'Quel est ton prénom?', "Comment tu t'appelles?", 'Qui es-tu?', 'bye', 'Salut', 'see ya', 'adios', 'cya']

print(doc_y)

['greeting', 'greeting', 'greeting', 'greeting', 'greeting', 'age', 'age', 'age', 'date', 'date', 'date', 'name', 'name', 'name', 'goodbye', 'goodbye', 'goodbye', 'goodbye', 'goodbye']

training = []
out_empty = [0] * len(classes)

for idx, doc in enumerate(doc_x):
    bow = []
    text = lemmatizer.lemmatize(doc.lower())
    for word in words:
        bow.append(1) if word in text else bow.append(0)

    output_row = list(out_empty)
    output_row[classes.index(doc_y[idx])] = 1

    training.append([bow, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)

train_x = np.array(list(training[:, 0]))
train_y = np.array(list(training[:, 1]))

input_shape = (len(train_x[0]),)
output_shape = len(train_y[0])
epochs = 200

model = Sequential()
model.add(Dense(128, input_shape=input_shape, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(output_shape, activation = "softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

print(model.summary())

model.fit(x=train_x, y=train_y, epochs=200,verbose=1)

def clean_text(text): 
  tokens = nltk.word_tokenize(text)
  tokens = [lemmatizer.lemmatize(word) for word in tokens]
  return tokens
def bag_of_words(text, vocab): 
  tokens = clean_text(text)
  bow = [0] * len(vocab)
  for w in tokens: 
    for idx, word in enumerate(vocab):
      if word == w: 
        bow[idx] = 1
  return np.array(bow)
def pred_class(text, vocab, labels): 
  bow = bag_of_words(text, vocab)
  result = model.predict(np.array([bow]))[0]
  thresh = 0.2
  y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
  y_pred.sort(key=lambda x: x[1], reverse=True)
  return_list = []
  for r in y_pred:
    return_list.append(labels[r[0]])
  return return_list
def get_response(intents_list, intents_json): 
  tag = intents_list[0]
  list_of_intents = intents_json["intents"]
  for i in list_of_intents: 
    if i["tag"] == tag:
      result = random.choice(i["responses"])
      break
  return result

while True:
    message = input("")
    intents = pred_class(message, words, classes)
    result = get_response(intents, data)
    print(result)