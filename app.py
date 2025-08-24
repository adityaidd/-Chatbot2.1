# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 10:46:51 2024
@author: Jay kumar gupta
"""

from flask_cors import CORS

import sys
sys.excepthook = lambda type, value, traceback: print(f"Uncaught exception: {value}")

import nltk
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import json
import random
from trainchatboat import train_chatbot 

print("Loading chatbot components...")
try:
    model = load_model('chatbot_model.h5')
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

try:
    with open('intent.json', 'r', encoding="utf8") as f:
        intents = json.load(f)
    print("Intents loaded successfully")
except Exception as e:
    print(f"Error loading intents: {e}")
    sys.exit(1)

try:
    with open('words.pkl', 'rb') as f:
        words = pickle.load(f)
    with open('classes.pkl', 'rb') as f:
        classes = pickle.load(f)
    print("Pickle files loaded successfully")
except Exception as e:
    print(f"Error loading pickle files: {e}")
    sys.exit(1)

print("All chatbot components loaded successfully.")

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.1
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    if ints:
        tag = ints[0]['intent']
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
    else:
        result = "I'm sorry, I didn't understand that. Could you please rephrase?"
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

from flask import Flask, jsonify, request

app = Flask(__name__)

CORS(app)



@app.route("/", methods=['GET', 'POST'])
def home():
    return jsonify({"key": "home page value"})

def decrypt(msg):
    return msg.replace("+", " ")


@app.route('/home/update', methods=['POST'])
def recompile_bot():
    print("hey")

    data = request.get_json()
    
    pro_city = data.get('pro_city')
    pro_locality = data.get('pro_locality')
    pro_amt = data.get('pro_amt')
    pro_area_size = data.get('pro_area_size')
    pro_desc = data.get('pro_desc')
    link=data.get('link')

    # Process the data as needed
    patterns = [
        f"Tell me about a property in {pro_city}",
        f"What are the details of the property in {pro_locality}?",
        f"How much does the property in {pro_city} cost?",
        f"Describe the property located in {pro_locality}",
        f"What is the area size of the property in {pro_city}?",
        f"Give me details about a {pro_area_size} sqft property in {pro_city}",
        f"How many bedrooms in the property located at {pro_locality}?",
        f"Is there a property in {pro_city} with a budget of {pro_amt}?",
        f"Details of a property in {pro_locality} with {pro_amt} price",
        f"Can you provide the description of a property in {pro_city}?"
    ]
    intent_data = {
        "tag": "property",
        "patterns": patterns,
        "responses": [
            "Here are the details of the property:",
            f"City: {pro_city}",
            f"Locality: {pro_locality}",
            f"Price: {pro_amt}",
            f"Area Size: {pro_area_size} sqft",
            f"Description: {pro_desc}"
        ],
        "context": [f"This is link from which you can search the property :{link}"]
    }
    with open('intent.json', 'r+', encoding='utf-8') as file:
        data = json.load(file)
        data['intents'].append(intent_data)
        file.seek(0)  # Move the cursor to the beginning of the file
        json.dump(data, file, indent=4)
        file.truncate() 
    train_chatbot()
    return jsonify(data), 200

@app.route('/home', methods=['GET'])
def home_page():
    name = request.args.get('name', '')
    dec_msg = decrypt(name)
    response = chatbot_response(dec_msg)
    json_obj = jsonify({"top": {"res": response}})
    return json_obj

@app.errorhandler(Exception)
def handle_error(e):
    print(f"An error occurred: {e}")
    return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=False)
