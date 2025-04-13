import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi PySastrawi ---
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# --- Fungsi Preprocessing ---
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    clean_text = ' '.join(tokens)
    no_stopword = stopword_remover.remove(clean_text)
    stemmed = stemmer.stem(no_stopword)
    return stemmed

# --- Load dan ambil hanya bagian 'data' dari file JSON ---
with open('repository_data.json', 'r', encoding='utf-8') as f:
    json_file = json.load(f)
    data = json_file["data"]

df = pd.DataFrame(data)
df['preprocessed'] = df['title'].apply(preprocess)

# --- TF-IDF dan Cosine Similarity ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])

# --- Fungsi pencarian dengan hasil TOP 5 ---
def search_repository(query, top_n=5):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[-top_n:][::-1]  # Urut dari yang paling mirip
    results = df.iloc[top_indices]

    output = []
    for _, row in results.iterrows():
        output.append({
            "title": row["title"],
            "link": row["link"]
        })

    return output

# --- Jalankan Program ---
if __name__ == "__main__":
    query = input("Apa yang ingin Anda cari?\n> ")
    results = search_repository(query)

    print("\nBerikut adalah beberapa judul yang relevan dengan keyword Anda:\n")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   Link: {result['link']}\n")
