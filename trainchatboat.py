import nltk
import json
import pickle
import numpy as np
import random
import tensorflow as tf
from tqdm import tqdm
import os

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def train_chatbot():
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!', '@', '$']

    data_file = open('intent.json', encoding="utf8").read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])
                
    if os.path.exists('words.pkl') and os.path.exists('classes.pkl'):
        words = pickle.load(open('words.pkl', 'rb'))
        classes = pickle.load(open('classes.pkl', 'rb'))
    else:
        words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        pickle.dump(words, open('words.pkl', 'wb'))
        pickle.dump(classes, open('classes.pkl', 'wb'))

    training = []
    output_empty = [0] * len(classes)
    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(w.lower()) for w in pattern_words]
        
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)
        
        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = np.array([item[0] for item in training])
    train_y = np.array([item[1] for item in training])

    model = tf.keras.Sequential([
        tf.keras.layers.Dense(256, input_shape=(len(train_x[0]),), activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(len(train_y[0]), activation='softmax')
    ])

    model.compile(loss='categorical_crossentropy', 
                  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
                  metrics=['accuracy'])

    print("Training model...")
    epochs = 500
    batch_size = 32

    history = model.fit(
        train_x, train_y,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_split=0.1
    )

    for epoch in tqdm(range(epochs), desc="Training Progress"):
        loss, accuracy = model.evaluate(train_x, train_y, verbose=0)
        if (epoch + 1) % 50 == 0:
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(f"Loss: {loss:.4f}")
            print(f"Accuracy: {accuracy:.4f}")

    model.save('chatbot_model.h5')
    print("Model created successfully")

    return model, words, classes

# You can call this function from app.py when recompile function is called
# model, words, classes = train_chatbot()