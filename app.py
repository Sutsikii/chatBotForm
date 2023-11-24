from flask import Flask, render_template, request
import os
import json
import string
import random
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from keras import Sequential
from keras.layers import Dense, Dropout
from unidecode import unidecode

nltk.download("punkt")
nltk.download("wordnet")

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()

words = []
classes = []
doc_x = []
doc_y = []

with open('data.json', 'r') as file:
    data = json.load(file)

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
model.add(Dense(output_shape, activation="softmax"))

adam = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=["accuracy"])

model.fit(x=train_x, y=train_y, epochs=200, verbose=1)

def pred_class(text, vocab, labels):
    bow = [0] * len(vocab)
    tokens = nltk.word_tokenize(text)
    tokens = [unidecode(word) for word in tokens]
    tokens = [lemmatizer.lemmatize(word.lower()) for word in tokens]
    for w in tokens:
        if w in vocab:
            bow[vocab.index(w)] = 1

    result = model.predict(np.array([bow]))[0]
    thresh = 0.2
    y_pred = [[idx, res] for idx, res in enumerate(result) if res > thresh]
    y_pred.sort(key=lambda x: x[1], reverse=True)
    return_list = [labels[r[0]] for r in y_pred]
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i["tag"] == tag:
            result = random.choice(i["responses"])
            break
    return result

def get_file_content(filename):
    with open(filename, 'r') as file:
        content = file.read()
    return content

@app.route('/')
def home():
    with open('data.json', 'r') as file:
        data_json_content = file.read()
    return render_template('index.php', data_json_content=data_json_content)

@app.route('/get')
def get_bot_response():
    user_text = request.args.get('msg')
    intents = pred_class(user_text, words, classes)
    for intent in data["intents"]:
        if intents and intent["tag"] == intents[0] and "file" in intent:
            file_content = get_file_content(intent["file"])
            return f'<pre>{file_content}</pre>'
    
    result = get_response(intents, data)
    return result

if __name__ == '__main__':
    app.run(debug=True)
