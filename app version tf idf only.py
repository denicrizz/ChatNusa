from sklearn.feature_extraction.text import TfidfVectorizer
import json
import pandas as pd
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stemmer dan stopword
stemmer = StemmerFactory().create_stemmer()
stopword = StopWordRemoverFactory().create_stop_word_remover()

def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    cleaned = ' '.join(tokens)
    no_stop = stopword.remove(cleaned)
    stemmed = stemmer.stem(no_stop)
    return stemmed

# Load data JSON
with open('repository_data.json', 'r', encoding='utf-8') as f:
    json_file = json.load(f)
    data = json_file['data']

df = pd.DataFrame(data)
df['preprocessed'] = df['title'].apply(preprocess)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
feature_names = vectorizer.get_feature_names_out()


def search_tf_only(query, top_n=10):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()

    scores = []
    for i, doc in enumerate(df['preprocessed']):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        doc_score = 0.0
        for token in query_tokens:
            if token in feature_names:
                idx = vectorizer.vocabulary_.get(token)
                if idx is not None:
                    doc_score += doc_vec[idx]
        scores.append((i, doc_score))

    # Urutkan berdasarkan skor tertinggi
    scores = sorted(scores, key=lambda x: x[1], reverse=True)[:top_n]
    results = [df.iloc[i] for i, _ in scores if _ > 0]

    output = []
    for result in results:
        output.append({
            "title": result["title"],
            "link": result["link"]
        })

    return output

# Program utama
if __name__ == "__main__":
    query = input("Apa yang ingin Anda cari?\n> ")
    results = search_tf_only(query)
    if results:
        print("\nBerikut adalah beberapa judul yang relevan dengan keyword Anda:\n")
        for i, r in enumerate(results, 1):
            print(f"{i}. {r['title']}\n   Link: {r['link']}\n")
    else:
        print("\nMaaf, tidak ditemukan judul yang relevan.")
