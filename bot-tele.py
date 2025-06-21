from flask import Flask, request
import json
import requests
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk
from nltk.corpus import stopwords
import string

# Download stopwords saat pertama kali
nltk.download('stopwords')

app = Flask(__name__)

TOKEN = "7352232743:AAEuC0nMxQWEpMoglvGMob4Vl5TaUjmIJRg"
URL = f"https://api.telegram.org/bot{TOKEN}/sendMessage"

# --- Inisialisasi ---
stemmer = StemmerFactory().create_stemmer()
stop_words = set(stopwords.words('indonesian'))

# --- Fungsi Preprocessing ---
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [t.strip(string.punctuation) for t in tokens if t.isalpha()]
    tokens = [t for t in tokens if t not in stop_words]
    cleaned = ' '.join(tokens)
    stemmed = stemmer.stem(cleaned)
    return stemmed

# --- Load repository_data1.json ---
with open("repository_data1.json", "r", encoding="utf-8") as f:
    repo = json.load(f)['data']
df_repo = pd.DataFrame(repo)
df_repo['preprocessed'] = df_repo['title'].apply(preprocess)
vectorizer_repo = TfidfVectorizer()
tfidf_matrix_repo = vectorizer_repo.fit_transform(df_repo['preprocessed'])
feature_names_repo = vectorizer_repo.get_feature_names_out()

# --- Load info_unp.json ---
with open("info_unp.json", "r", encoding="utf-8") as f:
    info_data = json.load(f)['data']
df_info = pd.DataFrame(info_data)
df_info['preprocessed'] = df_info['pertanyaan'].apply(preprocess)
vectorizer_info = TfidfVectorizer()
tfidf_matrix_info = vectorizer_info.fit_transform(df_info['preprocessed'])
feature_names_info = vectorizer_info.get_feature_names_out()

# --- Fungsi Pencarian Skripsi ---
def search_repository(query):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()
    results = []
    for i, doc in enumerate(df_repo['preprocessed']):
        doc_vec = tfidf_matrix_repo[i].toarray().flatten()
        score = sum(doc_vec[vectorizer_repo.vocabulary_.get(token, 0)] for token in query_tokens if token in feature_names_repo)
        if score > 0:
            results.append((i, score))
    results.sort(key=lambda x: x[1], reverse=True)
    output = []
    for i, _ in results[:5]:
        row = df_repo.iloc[i]
        output.append(f"📌 {row['title']}\n🔗 {row['link']}")
    return "\n\n".join(output) if output else None

# --- Fungsi Pencarian Info UNP ---
def search_info(query):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()
    results = []
    for i, doc in enumerate(df_info['preprocessed']):
        doc_vec = tfidf_matrix_info[i].toarray().flatten()
        score = sum(doc_vec[vectorizer_info.vocabulary_.get(token, 0)] for token in query_tokens if token in feature_names_info)
        if score > 0:
            results.append((i, score))
    results.sort(key=lambda x: x[1], reverse=True)
    output = []
    for i, _ in results[:3]:
        row = df_info.iloc[i]
        output.append(f"❓ {row['pertanyaan']}\n✅ {row['jawaban']}")
    return "\n\n".join(output) if output else None

# --- Endpoint Telegram Bot ---
@app.route("/", methods=["POST"])
def index():
    data = request.get_json()
    chat_id = data["message"]["chat"]["id"]
    text = data["message"].get("text", "")

    result_info = search_info(text)
    result_repo = search_repository(text)

    if result_info:
        response_text = "📚 *Informasi Kampus:*\n\n" + result_info
    elif result_repo:
        response_text = "📖 *Hasil Pencarian Skripsi:*\n\n" + result_repo
    else:
        response_text = "❌ Maaf, tidak ditemukan informasi yang relevan."

    requests.post(URL, json={"chat_id": chat_id, "text": response_text, "parse_mode": "Markdown"})
    return "ok"

if __name__ == "__main__":
    app.run(debug=True)
