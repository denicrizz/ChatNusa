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


# --- Load JSON dari repository dan info_unp ---
def load_data():
    all_data = []

    # Load repository_data1.json
    with open('repository_data1.json', 'r', encoding='utf-8') as f:
        repo_json = json.load(f)
        for item in repo_json:
            for entry in item["data"]:
                all_data.append({
                    "title": entry["title"],
                    "link": entry["link"],
                    "source": "repository"
                })

    # Load info_unp.json (pakai kunci: pertanyaan dan jawaban)
    with open('info_unp.json', 'r', encoding='utf-8') as f:
        unp_json = json.load(f)
        for entry in unp_json:
            all_data.append({
                "title": entry["pertanyaan"],  # <- disesuaikan
                "link": entry["jawaban"],      # <- disesuaikan
                "source": "info_unp"
            })

    return pd.DataFrame(all_data)


# --- Siapkan Data ---
df = load_data()
df['preprocessed'] = df['title'].apply(preprocess)
print(f"Total data dimuat: {len(df)}")

# --- TF-IDF ---
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df['preprocessed'])
feature_names = vectorizer.get_feature_names_out()

# --- Fungsi Pencarian ---
def search_repository(query, top_repo=5, top_unp=1):
    query_processed = preprocess(query)
    query_tokens = query_processed.split()
    vocab_index = {token: i for i, token in enumerate(feature_names)}
    query_indices = [vocab_index[token] for token in query_tokens if token in vocab_index]

    scores = []
    for i in range(tfidf_matrix.shape[0]):
        row = tfidf_matrix[i]
        score = sum(row[0, idx] for idx in query_indices)
        scores.append(score)

    df_copy = df.copy()
    df_copy['score'] = scores
    sorted_df = df_copy.sort_values(by='score', ascending=False).drop_duplicates(subset='title')

    # Pisahkan hasil berdasarkan sumber
    unp_results = sorted_df[sorted_df['source'] == 'info_unp']
    repo_results = sorted_df[sorted_df['source'] == 'repository']

    # Jika ada hasil relevan dari info_unp, tampilkan hanya itu
    if not unp_results.empty and unp_results['score'].iloc[0] > 0:
        unp_results = unp_results.head(top_unp)
        final_results = unp_results
    else:
        # Jika tidak ada, tampilkan dari repository
        final_results = repo_results.head(top_repo)

    output = []
    for _, row in final_results.iterrows():
        output.append({
            "title": row["title"],
            "link": row["link"],
            "source": row["source"]
        })

    return output


# --- Jalankan Program ---
if __name__ == "__main__":
    query = input("Apa yang ingin Anda cari?\n> ")
    results = search_repository(query)

    print("\nHasil pencarian Anda:\n")
    for i, result in enumerate(results, start=1):
        sumber = "📚 Repository" if result["source"] == "repository" else "🏫 Info UNP"
        print(f"{i}. {result['title']} ({sumber})")
        print(f"   Link: {result['link']}\n")
