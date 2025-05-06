import re
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# Inisialisasi stemmer dan stopword
stemmer_factory = StemmerFactory()
stemmer = stemmer_factory.create_stemmer()
stopword_factory = StopWordRemoverFactory()
stopword = stopword_factory.create_stop_word_remover()

# Tambahkan stopword kustom untuk meningkatkan relevansi
default_stopwords = stopword_factory.get_stop_words()
custom_stopwords = ['atau', 'dengan', 'yang', 'dan', 'di', 'ke', 'dari', 'untuk']
all_stopwords = list(set(default_stopwords + custom_stopwords))

# Cara yang benar untuk menggunakan stopword kustom
stopword_factory = StopWordRemoverFactory()
# Set stopword kustom ke factory
stopword_factory.get_stop_words = lambda: all_stopwords
# Buat stopword remover dengan stopword yang sudah dikustomisasi
custom_stopword_remover = stopword_factory.create_stop_word_remover()

def preprocess_improved(text):
    if not isinstance(text, str):
        return ""
    
    # Normalisasi
    text = text.lower()
    
    # Hapus URL
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Hapus tanda baca dan karakter khusus
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Hapus angka
    text = re.sub(r'\d+', '', text)
    
    # Hapus spasi berlebih
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Stemming terlebih dahulu
    stemmed_text = stemmer.stem(text)
    
    # Hapus stopword setelah stemming
    cleaned_text = custom_stopword_remover.remove(stemmed_text)
    
    # Tokenisasi hasil akhir
    tokens = cleaned_text.split()
    
    # Filter token kosong dan token pendek (kurang dari 3 karakter)
    tokens = [token for token in tokens if token and len(token) >= 3]
    
    # Gabungkan kembali
    final_text = ' '.join(tokens)
    
    return final_text

# Load data Info UNP
try:
    with open('info_unp.json', 'r', encoding='utf-8') as f:
        info_data = json.load(f)
    
    info_df = pd.DataFrame(info_data)
    # Pastikan kolom yang diperlukan ada
    if 'pertanyaan' not in info_df.columns or 'jawaban' not in info_df.columns:
        print("⚠️ Format data info_unp.json tidak sesuai. Pastikan ada kolom 'pertanyaan' dan 'jawaban'.")
    info_df['preprocessed'] = info_df['pertanyaan'].apply(preprocess_improved)
except Exception as e:
    print(f"⚠️ Error saat memuat info_unp.json: {str(e)}")
    info_df = pd.DataFrame(columns=['pertanyaan', 'jawaban', 'preprocessed'])

# Load data Repository
try:
    with open('repository_data1.json', 'r', encoding='utf-8') as f:
        repo_json = json.load(f)
        # Format JSON adalah array yang berisi satu objek dengan kunci 'data'
        repo_data = repo_json[0]['data']
    
    repo_df = pd.DataFrame(repo_data)
    # Pastikan kolom yang diperlukan ada
    if 'title' not in repo_df.columns or 'link' not in repo_df.columns:
        print("⚠️ Format data repository_data1.json tidak sesuai. Pastikan ada kolom 'title' dan 'link'.")
    repo_df['preprocessed'] = repo_df['title'].apply(preprocess_improved)
except Exception as e:
    print(f"⚠️ Error saat memuat repository_data1.json: {str(e)}")
    repo_df = pd.DataFrame(columns=['title', 'link', 'year', 'authors', 'source', 'preprocessed'])

# Buat vektor TF-IDF hanya jika data tersedia
if not info_df.empty:
    info_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    info_tfidf = info_vectorizer.fit_transform(info_df['preprocessed'])
    info_features = info_vectorizer.get_feature_names_out()
else:
    info_vectorizer = None
    info_tfidf = None
    info_features = []

if not repo_df.empty:
    repo_vectorizer = TfidfVectorizer(min_df=1, ngram_range=(1, 2))
    repo_tfidf = repo_vectorizer.fit_transform(repo_df['preprocessed'])
    repo_features = repo_vectorizer.get_feature_names_out()
else:
    repo_vectorizer = None
    repo_tfidf = None
    repo_features = []

def calculate_score_improved(query_tokens, tfidf_matrix, features, vectorizer):
    if tfidf_matrix is None or len(query_tokens) == 0:
        return []
    
    scores = []
    for i in range(tfidf_matrix.shape[0]):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        doc_score = 0.0
        
        # Hitung skor berdasarkan token yang cocok
        token_matches = 0
        for token in query_tokens:
            if token in features:
                idx = vectorizer.vocabulary_.get(token)
                if idx is not None:
                    doc_score += doc_vec[idx]
                    token_matches += 1
        
        # Bobot tambahan jika persentase token yang cocok tinggi
        if len(query_tokens) > 0:
            match_percentage = token_matches / len(query_tokens)
            doc_score *= (1 + match_percentage)
        
        scores.append((i, doc_score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    return scores

def get_authors_formatted(authors_list):
    if not authors_list:
        return "Tidak ada data penulis"
    
    if len(authors_list) == 1:
        return authors_list[0]
    elif len(authors_list) == 2:
        return f"{authors_list[0]} dan {authors_list[1]}"
    else:
        return f"{authors_list[0]}, {authors_list[1]}, dkk."

def search_system_improved(query, threshold=0.1):
    # Preprocess query
    query_pre = preprocess_improved(query)
    query_tokens = query_pre.split()
    
    if not query_tokens:
        return "none", None, "Query terlalu pendek atau hanya berisi stopwords"
    
    # Hitung skor Info UNP
    info_scores = calculate_score_improved(query_tokens, info_tfidf, info_features, info_vectorizer)
    best_info_idx, best_info_score = info_scores[0] if info_scores else (-1, 0)
    
    # Hitung skor Repository
    repo_scores = calculate_score_improved(query_tokens, repo_tfidf, repo_features, repo_vectorizer)
    
    # Filter skor repo yang di atas threshold
    top_repo_scores = [(idx, score) for idx, score in repo_scores if score > threshold][:5]
    
    # Bandingkan skor terbaik dari kedua sumber
    best_repo_score = top_repo_scores[0][1] if top_repo_scores else 0
    
    if best_info_score >= best_repo_score and best_info_score > threshold:
        # Kembalikan hasil dari info UNP
        result = info_df.iloc[best_info_idx]
        return "info", {
            "pertanyaan": result["pertanyaan"],
            "jawaban": result["jawaban"],
            "score": best_info_score
        }, None
    elif top_repo_scores:
        # Kembalikan beberapa hasil teratas dari repository
        results = []
        for idx, score in top_repo_scores:
            row = repo_df.iloc[idx]
            # Pastikan semua kolom ada atau gunakan nilai default
            authors = row.get("authors", [])
            authors_formatted = get_authors_formatted(authors)
            
            results.append({
                "title": row["title"],
                "link": row["link"],
                "year": row.get("year", "Tidak diketahui"),
                "authors": authors_formatted,
                "source": row.get("source", ""),
                "score": score
            })
        return "repository", results, None
    else:
        return "none", None, "Tidak ditemukan hasil yang relevan"

# Program utama
if __name__ == "__main__":
    print("🔍 Sistem Pencarian Repositori dan Informasi UNP")
    print("================================================")
    
    while True:
        query = input("\nApa yang ingin Anda cari? (ketik 'exit' untuk keluar)\n> ")
        
        if query.lower() == 'exit':
            print("\n👋 Terima kasih telah menggunakan sistem pencarian.")
            break
            
        # Threshold minimal untuk menganggap hasil relevan
        result_type, result, message = search_system_improved(query, threshold=0.1)
        
        if result_type == "info":
            print("\n✅ Berikut informasi yang saya temukan untuk Anda:")
            print(f"\n❓ Pertanyaan: {result['pertanyaan']}")
            print(f"\n💬 Jawaban: {result['jawaban']}")
            print(f"\n📊 Skor relevansi: {result['score']:.4f}\n")
        
        elif result_type == "repository":
            print(f"\n📚 Berikut {len(result)} repository yang mungkin Anda cari:\n")
            
            for i, repo in enumerate(result, 1):
                print(f"Result #{i} (Skor: {repo['score']:.4f})")
                print(f"📝 Judul: {repo['title']}")
                print(f"👤 Penulis: {repo['authors']}")
                print(f"📅 Tahun: {repo['year']}")
                print(f"🔗 Link: {repo['link']}")
                print(f"📄 Sumber: {repo['source']}")
                print("-" * 50)
        
        else:
            print(f"\n❌ {message}")