import joblib
from sklearn.metrics.pairwise import cosine_similarity
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# --- Inisialisasi ---
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

# --- Custom stemmer & preprocessing ---
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

# --- Load model ---
model_data = joblib.load('ChatNusa-v1.joblib')
df = model_data['data']
vectorizer = model_data['vectorizer']
doc_vectors = model_data['doc_vectors']

# --- Search function ---
def search(query, top_repo=5, top_unp=1):
    query_processed = preprocess(query)
    query_vector = vectorizer.transform([query_processed])

    similarity_scores = cosine_similarity(query_vector, doc_vectors).flatten()

    df_copy = df.copy()
    df_copy['score'] = similarity_scores
    sorted_df = df_copy.sort_values(by='score', ascending=False).drop_duplicates(subset='title')

    unp_results = sorted_df[sorted_df['source'] == 'info_unp']
    repo_results = sorted_df[sorted_df['source'] == 'repository']

    if not unp_results.empty and unp_results['score'].iloc[0] > 0:
        final_results = unp_results.head(top_unp)
    else:
        final_results = repo_results.head(top_repo)

    return final_results

# --- CLI Interface ---
if __name__ == "__main__":
    print("🔍 Pencarian dimulai...")
    while True:
        q = input("\nKetik pertanyaanmu (atau 'exit' untuk keluar):\n> ")
        if q.lower() == 'exit':
            break

        results = search(q)
        print("\nHasil:")
        for i, row in enumerate(results.itertuples(), 1):
            sumber = "📚 Repository" if row.source == "repository" else "🏫 Info UNP"
            print(f"{i}. {row.title} ({sumber})")
            print(f"   Link: {row.link}\n")
