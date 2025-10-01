import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import NearestCentroid
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import random
import time
import joblib
from textblob import TextBlob
from collections import Counter

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Initialize lemmatizer and stop words
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stop words and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def generate_dataset_from_json(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    questions = []
    for intent in data['intents']:
        for pattern in intent['patterns']:
            questions.append(pattern)
    
    return questions

# Generate dataset from intent.json
questions = generate_dataset_from_json('intent.json')
print(f"Dataset size: {len(questions)}")

# Preprocess the data
print("Preprocessing data...")
preprocessed_questions = [preprocess_text(q) for q in questions]

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(preprocessed_questions)
print(f"Vector shape: {X.shape}")

# Dimensionality reduction
print("Performing dimensionality reduction...")
svd = TruncatedSVD(n_components=100)
X_reduced = svd.fit_transform(X)
X_normalized = normalize(X_reduced)

# Function to find optimal number of clusters
def find_optimal_clusters(X, max_k):
    silhouette_scores = []
    for k in range(2, max_k+1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"Silhouette score for k={k}: {score}")
    return silhouette_scores

# Find optimal number of clusters
print("Finding optimal number of clusters...")
max_k = 100  # Reduced for demonstration, you can adjust as needed
silhouette_scores = find_optimal_clusters(X_normalized, max_k)

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(range(2, max_k+1), silhouette_scores, 'bo-')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score vs Number of Clusters')
plt.savefig('silhouette_scores.png')
plt.show()
plt.close()

# Choose the number of clusters with the highest silhouette score
optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Optimal number of clusters: {optimal_k}")

# Perform clustering using different methods
print("Performing clustering...")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_clusters = kmeans.fit_predict(X_normalized)

agglomerative = AgglomerativeClustering(n_clusters=optimal_k)
agglomerative_clusters = agglomerative.fit_predict(X_normalized)

gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_clusters = gmm.fit_predict(X_normalized)

# Train a NearestCentroid classifier on Agglomerative Clustering results
agglomerative_classifier = NearestCentroid()
agglomerative_classifier.fit(X_normalized, agglomerative_clusters)

# Function to get top words for each cluster
def get_top_words(vectorizer, kmeans, n_words=10):
    feature_names = vectorizer.get_feature_names_out()
    top_words = {}
    for i in range(kmeans.n_clusters):
        center = kmeans.cluster_centers_[i]
        top_word_indices = center.argsort()[-n_words:][::-1]
        top_words[i] = [feature_names[ind] for ind in top_word_indices]
    return top_words

top_words = get_top_words(vectorizer, kmeans)
print("\nTop words for each cluster:")
for cluster, words in top_words.items():
    print(f"Cluster {cluster}: {', '.join(words)}")

# Generate more sophisticated responses for each cluster
def generate_response(cluster, top_words):
    words = top_words[cluster][:100]
    responses = [
        f"I see you're interested in {', '.join(words)}. How can I assist you with that?",
        f"It seems like you're asking about {', '.join(words)}. What specific information are you looking for?",
        f"Ah, {', '.join(words)}! That's an interesting topic. What would you like to know?",
        f"I have some information about {', '.join(words)}. What aspect are you curious about?",
        f"When it comes to {', '.join(words)}, there's a lot to discuss. Where should we start?"
    ]
    return random.choice(responses)

responses = {i: generate_response(i, top_words) for i in range(optimal_k)}

# Save the models and vectorizer
joblib.dump(kmeans, 'kmeans_model.joblib')
joblib.dump(agglomerative_classifier, 'agglomerative_classifier.joblib')
joblib.dump(gmm, 'gmm_model.joblib')
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
joblib.dump(svd, 'svd_model.joblib')
joblib.dump(responses, 'cluster_responses.joblib')
print("Models, vectorizer, and responses saved.")

# Sentiment analysis
def analyze_sentiment(text):
    return TextBlob(text).sentiment.polarity

# Train a simple classifier for intent recognition
intents = [intent['tag'] for intent in json.load(open('intent.json', encoding='utf-8'))['intents']]  # Specify encoding as 'utf-8'
X_train, X_test, y_train, y_test = train_test_split(X, [intents[i % len(intents)] for i in range(len(questions))], test_size=0.2, random_state=42)
intent_classifier = SVC(kernel='linear')
intent_classifier.fit(X_train, y_train)
joblib.dump(intent_classifier, 'intent_classifier.joblib')

# Function to get response based on user query
def get_response(query):
    start_time = time.time()
    preprocessed_query = preprocess_text(query)
    query_vector = vectorizer.transform([preprocessed_query])
    query_vector_reduced = svd.transform(query_vector)
    query_vector_normalized = normalize(query_vector_reduced)
    
    kmeans_cluster = kmeans.predict(query_vector_normalized)[0]
    agglomerative_cluster = agglomerative_classifier.predict(query_vector_normalized)[0]
    gmm_cluster = gmm.predict(query_vector_normalized)[0]
    
    sentiment = analyze_sentiment(query)
    intent = intent_classifier.predict(query_vector)[0]
    
    response_time = time.time() - start_time
    print(f"Query: '{query}'")
    print(f"Preprocessed query: '{preprocessed_query}'")
    print(f"KMeans Cluster: {kmeans_cluster}")
    print(f"Agglomerative Cluster: {agglomerative_cluster}")
    print(f"GMM Cluster: {gmm_cluster}")
    print(f"Sentiment: {sentiment:.2f}")
    print(f"Intent: {intent}")
    print(f"Response time: {response_time:.4f} seconds")
    
    response = responses[kmeans_cluster]
    if sentiment > 0.5:
        response += " I'm glad you're feeling positive about this!"
    elif sentiment < -0.5:
        response += " I understand this might be a sensitive topic."
    
    return response

# Main loop for interaction
print("\nChatbot is ready. Enter your queries or type 'exit' to end the session.")
chat_history = []

while True:
    query = input("You: ")
    if query.lower() == 'exit':
        break
    response = get_response(query)
    print(f"Bot: {response}\n")
    chat_history.append((query, response))

print("Chat session ended.")

# Analyze chat history
if chat_history:
    print("\nChat History Analysis:")
    total_queries = len(chat_history)
    avg_query_length = sum(len(q) for q, _ in chat_history) / total_queries
    avg_response_length = sum(len(r) for _, r in chat_history) / total_queries
    
    print(f"Total queries: {total_queries}")
    print(f"Average query length: {avg_query_length:.2f} characters")
    print(f"Average response length: {avg_response_length:.2f} characters")
    
    # Word frequency analysis
    all_words = ' '.join([q for q, _ in chat_history]).split()
    word_freq = Counter(all_words)
    print("\nTop 10 most frequent words in queries:")
    for word, count in word_freq.most_common(10):
        print(f"{word}: {count}")

    # Plot query lengths
    plt.figure(figsize=(10, 6))
    plt.hist([len(q) for q, _ in chat_history], bins=20)
    plt.xlabel('Query Length (characters)')
    plt.ylabel('Frequency')
    plt.title('Distribution of Query Lengths')
    plt.savefig('query_length_distribution.png')
    plt.show()
    plt.close()

# Visualize clusters using scatter plot
def plot_clusters(X_reduced, labels, title):
    plt.figure(figsize=(10, 8))
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.5, s=50)
    plt.title(title)
    plt.colorbar()
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig(f'{title}.png')
    plt.show()
    plt.close()

plot_clusters(X_reduced, kmeans_clusters, 'KMeans Clustering')
plot_clusters(X_reduced, agglomerative_clusters, 'Agglomerative Clustering')
plot_clusters(X_reduced, gmm_clusters, 'Gaussian Mixture Model Clustering')

print("\nChatbot session and analysis complete.")
