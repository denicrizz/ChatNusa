import json
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi Sastrawi ---
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

def preprocess(text):
    text = text.lower()
    tokens = text.split()
    tokens = [token for token in tokens if token.isalpha()]
    clean_text = ' '.join(tokens)
    no_stopword = stopword_remover.remove(clean_text)
    stemmed = custom_stemmer(no_stopword)
    return stemmed

# --- Load dan preprocess data ---
def load_data():
    all_data = []

    with open('repository_data1.json', 'r', encoding='utf-8') as f:
        repo_json = json.load(f)
        for item in repo_json:
            for entry in item["data"]:
                all_data.append({
                    "title": entry["title"],
                    "link": entry["link"],
                    "source": "repository"
                })

    with open('info_unp.json', 'r', encoding='utf-8') as f:
        unp_json = json.load(f)
        for entry in unp_json:
            all_data.append({
                "title": entry["pertanyaan"],
                "link": entry["jawaban"],
                "source": "info_unp"
            })

    return pd.DataFrame(all_data)

# --- Bangun model dan simpan ---
df = load_data()
df['preprocessed'] = df['title'].apply(preprocess)

vectorizer = CountVectorizer()
doc_vectors = vectorizer.fit_transform(df['preprocessed'])

model_data = {
    'data': df,
    'vectorizer': vectorizer,
    'doc_vectors': doc_vectors
}

joblib.dump(model_data, 'ChatNusa-v1.joblib')
print("✅ Model berhasil dibuat dan disimpan ke ChatNusa-v1.joblib")
