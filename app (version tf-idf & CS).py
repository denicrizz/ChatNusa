import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory


stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def custom_stemmer(text):

    kata_khusus = {
        "pengelasan": "ngelas",
        "pembelajaran": "ajar",
        "berbasis": "basis"
    }


    tokens = text.split()
    hasil = []

    for token in tokens:
        if token in kata_khusus:
            hasil.append(kata_khusus[token])
        else:
            hasil.append(stemmer.stem(token))

    return ' '.join(hasil)


def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    clean_text = ' '.join(tokens)
    no_stopword = stopword_remover.remove(clean_text)
    stemmed = custom_stemmer(no_stopword)

     #print(f"[DEBUG] Asli: {text}")
    #print(f"[DEBUG] Stemming: {stemmed}")

    return stemmed



# --- Load dan ambil hanya bagian 'data' dari file JSON ---
with open('repository_data1.json', 'r', encoding='utf-8') as f:
    json_file = json.load(f)
    
    # Gabungkan semua 'data' dari tiap elemen
    data = []
    for item in json_file:
        data.extend(item["data"])  # Tambahkan semua entri ke satu list


df = pd.DataFrame(data)
df['preprocessed'] = df['title'].apply(preprocess)

print(f"Jumlah data yang dibaca: {len(data)}")

# --- TF-IDF dan Cosine Similarity ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])

# --- Fungsi pencarian dengan hasil TOP 5 ---
def search_repository(query, top_n=5):
    query_processed = preprocess(query)
    query_vec = vectorizer.transform([query_processed])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_indices = similarities.argsort()[::-1]  # Ambil dari tertinggi ke rendah
    filtered_results = []

    for i in top_indices:
        title = df.iloc[i]['title']
        preprocessed_title = df.iloc[i]['preprocessed']

        if query_processed in preprocessed_title:
            filtered_results.append({
                "title": title,
                "link": df.iloc[i]["link"]
            })

        if len(filtered_results) >= top_n:
            break

    return filtered_results


# --- Jalankan Program ---
if __name__ == "__main__":
    query = input("Apa yang ingin Anda cari?\n> ")
    results = search_repository(query)

    print("\nBerikut adalah beberapa judul yang relevan dengan keyword Anda:\n")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   Link: {result['link']}\n")
        
