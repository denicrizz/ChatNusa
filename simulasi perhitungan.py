import numpy as np
from collections import Counter
import math
from tabulate import tabulate

def calculate_tf(document):
    word_freq = Counter(document)
    return {word: freq/len(document) for word, freq in word_freq.items()}

def calculate_idf(documents, term):
    doc_count = sum(1 for doc in documents if term in doc)
    if doc_count == 0:
        return 0
    return math.log(len(documents) / doc_count)

def calculate_tfidf(tf_dict, documents):
    tfidf_dict = {}
    for term, tf in tf_dict.items():
        idf = calculate_idf(documents, term)
        tfidf_dict[term] = tf * idf
    return tfidf_dict

def cosine_similarity(vec1, vec2):
    dot_product = sum(vec1[i] * vec2[i] for i in range(len(vec1)))
    mag1 = math.sqrt(sum(val**2 for val in vec1))
    mag2 = math.sqrt(sum(val**2 for val in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0
    return dot_product / (mag1 * mag2)

# Data
documents = [
    ["universitas", "nusantara", "sediakan", "layanan", "akademik"],
    ["layanan", "informasi", "akses", "mahasiswa"],
    ["mahasiswa", "peroleh", "informasi", "akademik"]
]

query = ["layanan", "informasi", "mahasiswa"]

print("=== DATA ===")
print("D1:", " ".join(documents[0]))
print("D2:", " ".join(documents[1]))
print("D3:", " ".join(documents[2]))
print("Query:", " ".join(query))
print("\n")

# Kumpulkan semua unique terms
all_terms = sorted(list(set(
    sum(documents, []) + query
)))

# Hitung TF untuk setiap dokumen dan query
tf_docs = [calculate_tf(doc) for doc in documents]
tf_query = calculate_tf(query)

# Hitung TF-IDF
tfidf_docs = [calculate_tfidf(tf_doc, documents) for tf_doc in tf_docs]
tfidf_query = calculate_tfidf(tf_query, documents)

# Tampilkan TF-IDF dalam bentuk tabel
print("=== HASIL TF-IDF ===")
headers = ["Term", "TF-IDF D1", "TF-IDF D2", "TF-IDF D3", "TF-IDF Query"]
rows = []
for term in all_terms:
    row = [
        term,
        round(tfidf_docs[0].get(term, 0), 3),
        round(tfidf_docs[1].get(term, 0), 3),
        round(tfidf_docs[2].get(term, 0), 3),
        round(tfidf_query.get(term, 0), 3)
    ]
    rows.append(row)

print(tabulate(rows, headers=headers, tablefmt="grid"))
print("\n")

# Konversi ke vektor dan hitung Cosine Similarity
def to_vector(tfidf_dict):
    return [tfidf_dict.get(term, 0) for term in all_terms]

similarities = []
query_vector = to_vector(tfidf_query)
for doc_tfidf in tfidf_docs:
    doc_vector = to_vector(doc_tfidf)
    sim = cosine_similarity(query_vector, doc_vector)
    similarities.append(sim)

# Tampilkan hasil Cosine Similarity
print("=== HASIL COSINE SIMILARITY ===")
results = []
for i, sim in enumerate(similarities, 1):
    results.append([f"D{i}", round(sim, 3)])

# Urutkan berdasarkan nilai similarity tertinggi
results.sort(key=lambda x: x[1], reverse=True)

print(tabulate(results, headers=["Dokumen", "Similarity"], tablefmt="grid"))
print("\nKESIMPULAN:")
print(f"Dokumen {results[0][0]} adalah yang paling relevan dengan query")
print(f"dengan nilai similarity tertinggi: {results[0][1]}")