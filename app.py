import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi ---
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# --- Custom Stemmer ---
def custom_stemmer(text):
    kata_khusus = {
        "pengelasan": "ngelas",
        "pembelajaran": "ajar",
        "berbasis": "basis"
    }
    tokens = text.split()
    hasil = [kata_khusus[token] if token in kata_khusus else stemmer.stem(token) for token in tokens]
    return ' '.join(hasil)

# --- Preprocessing ---
def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    clean_text = ' '.join(tokens)
    no_stopword = stopword_remover.remove(clean_text)
    stemmed = custom_stemmer(no_stopword)
    return stemmed

# --- Load JSON & Gabungkan ---
with open('repository_data1.json', 'r', encoding='utf-8') as f:
    json_file = json.load(f)
    data = []
    for item in json_file:
        data.extend(item["data"])

df = pd.DataFrame(data)
df['preprocessed'] = df['title'].apply(preprocess)
print(f"Jumlah data yang dibaca: {len(data)}")

# --- TF-IDF ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
feature_names = vectorizer.get_feature_names_out()

# --- Fungsi Pencarian dengan Pembobotan Berdasarkan Kemunculan Kata Kunci ---
def search_repository(query, top_n=5):
    query_processed = preprocess(query)
    query_tokens = query_processed.split()

    # Buat dictionary token -> index dalam TF-IDF matrix
    vocab_index = {token: i for i, token in enumerate(feature_names)}

    # Ambil indeks token dari query yang ada di vocab
    query_indices = [vocab_index[token] for token in query_tokens if token in vocab_index]

    # Hitung skor manual: jumlah bobot TF-IDF token dari query yang muncul
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        score = sum(row[0, idx] for idx in query_indices)
        scores.append(score)

    # Urutkan skor menurun
    sorted_indices = pd.Series(scores).argsort()[::-1]
    results = df.iloc[sorted_indices]

    # Hilangkan duplikat berdasarkan judul
    results = results.drop_duplicates(subset='title')
    results = results.head(top_n)

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

    print("\nBerikut adalah beberapa judul yang relevan dengan keyword Anda (berdasarkan TF-IDF yang diperkuat frekuensi token):\n")
    for i, result in enumerate(results, start=1):
        print(f"{i}. {result['title']}")
        print(f"   Link: {result['link']}\n")
