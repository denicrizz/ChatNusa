from flask import Flask, request
import json
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

app = Flask(__name__)

TOKEN = "7352232743:AAEuC0nMxQWEpMoglvGMob4Vl5TaUjmIJRg"
URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# --- Inisialisasi Sastrawi ---
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

# --- Preprocessing Function ---
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [t for t in tokens if t.isalpha()]
    cleaned = ' '.join(tokens)
    cleaned = stopword.remove(cleaned)
    stemmed = stemmer.stem(cleaned)
    return stemmed

# --- Load Data & TF-IDF ---
with open("repository_data.json", "r", encoding="utf-8") as f:
    repo = json.load(f)['data']

df = pd.DataFrame(repo)
df['preprocessed'] = df['title'].apply(preprocess)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
feature_names = vectorizer.get_feature_names_out()

# --- Pencarian Tanpa Cosine ---
def search_tf_only(query):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()
    results = []

    for i, doc in enumerate(df['preprocessed']):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        score = sum(doc_vec[vectorizer.vocabulary_.get(token, 0)] for token in query_tokens if token in feature_names)
        if score > 0:
            results.append((i, score))

    results.sort(key=lambda x: x[1], reverse=True)

    output = []
    for i, _ in results[:10]:
        row = df.iloc[i]
        output.append(f"📌 {row['title']}\n🔗 {row['link']}")
    return "\n\n".join(output) if output else "Maaf, tidak ditemukan judul yang relevan."

@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    response_text = "Berikut adalah hasil pencarian:\n\n" + search_tf_only(text)
    requests.post(URL, json={"chat_id": chat_id, "text": response_text})
    return "ok"


if __name__ == "__main__":
    app.run(debug=True)
