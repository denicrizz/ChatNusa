from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import matplotlib.pyplot as plt
import seaborn as sns

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

def evaluate_query(query, threshold=0.1):
    query_pre = preprocess(query)
    query_tokens = query_pre.split()

    predicted_relevance = []
    
    for i, doc in enumerate(df['preprocessed']):
        doc_vec = tfidf_matrix[i].toarray().flatten()
        doc_score = 0.0
        for token in query_tokens:
            if token in feature_names:
                idx = vectorizer.vocabulary_.get(token)
                if idx is not None:
                    doc_score += doc_vec[idx]
        
        # Dokumen dianggap relevan jika skor di atas threshold
        predicted_relevance.append(1 if doc_score > threshold else 0)

    return predicted_relevance

def evaluate_search(actual, predicted):
    # Hitung confusion matrix
    cm = confusion_matrix(actual, predicted)
    
    # Hitung metrik kinerja
    report = classification_report(actual, predicted)
    accuracy = accuracy_score(actual, predicted)
    
    # Tampilkan hasil evaluasi
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)
    
    # Visualisasi confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Tidak Relevan', 'Relevan'],
                yticklabels=['Tidak Relevan', 'Relevan'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()
    
    # Hitung metrik dari confusion matrix
    tn, fp, fn, tp = cm.ravel()
    
    # Precision, recall, dan F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\nMetrik dari Confusion Matrix:")
    print(f"True Positives (TP): {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"True Negatives (TN): {tn}")
    print(f"False Negatives (FN): {fn}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")

def create_ground_truth_labels():

    # Mensimulasikan dataset evaluasi (dalam kasus nyata, Anda akan memiliki ground truth)
    
    # Contoh penentuan relevansi berdasarkan kata kunci tertentu
    keywords = ["penelitian", "sistem", "analisis", "metode", "aplikasi", "perancangan"]
    
    # Menentukan dokumen yang relevan berdasarkan kata kunci
    relevance_labels = []
    for text in df['preprocessed']:
        is_relevant = any(keyword in text for keyword in keywords)
        relevance_labels.append(1 if is_relevant else 0)
    
    return relevance_labels

# Program utama untuk evaluasi
if __name__ == "__main__":
    print("Evaluasi Sistem Pencarian dengan Confusion Matrix")
    print("------------------------------------------------")
    
    # Dapatkan ground truth labels
    actual_labels = create_ground_truth_labels()
    
    # Informasi dataset
    relevant_count = sum(actual_labels)
    total_count = len(actual_labels)
    print(f"Dataset berisi {total_count} dokumen, {relevant_count} relevan dan {total_count - relevant_count} tidak relevan.")
    
    # Evaluasi query tunggal
    query = input("\nMasukkan query untuk evaluasi: ")
    threshold = float(input("Masukkan threshold (misalnya 0.1): "))
    
    predicted_labels = evaluate_query(query, threshold)
    
    print(f"\nEvaluasi untuk query: '{query}' dengan threshold {threshold}")
    evaluate_search(actual_labels, predicted_labels)
    
    # Evaluasi multi-query
    if input("\nApakah Anda ingin mengevaluasi beberapa query sekaligus? (y/n): ").lower() == 'y':
        queries = []
        while True:
            q = input("Masukkan query (kosongkan untuk berhenti): ")
            if not q:
                break
            queries.append(q)
        
        if queries:
            all_actual = []
            all_predicted = []
            
            # Evaluasi dan tampilkan hasil untuk setiap query
            for query in queries:
                pred = evaluate_query(query, threshold)
                
                print(f"\nEvaluasi untuk query: '{query}' dengan threshold {threshold}")
                evaluate_search(actual_labels, pred)
                
                # Kumpulkan data untuk evaluasi gabungan
                all_actual.extend(actual_labels)
                all_predicted.extend(pred)
            
            # Evaluasi gabungan semua query
            if len(queries) > 1:
                print("\n===== EVALUASI GABUNGAN SEMUA QUERY =====")
                evaluate_search(all_actual, all_predicted)
    
    # Evaluasi sensitivitas threshold
    if input("\nApakah Anda ingin mengevaluasi sensitivitas threshold? (y/n): ").lower() == 'y':
        test_query = input("Masukkan query untuk pengujian threshold: ")
        thresholds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
        
        results = []
        for thresh in thresholds:
            pred = evaluate_query(test_query, thresh)
            tn, fp, fn, tp = confusion_matrix(actual_labels, pred).ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            
            results.append({
                'threshold': thresh,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy
            })
        
        # Buat DataFrame untuk hasil
        results_df = pd.DataFrame(results)
        print("\nHasil evaluasi threshold:")
        print(results_df)
        
        # Visualisasi hasil threshold
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['threshold'], results_df['precision'], 'b-', label='Precision')
        plt.plot(results_df['threshold'], results_df['recall'], 'g-', label='Recall')
        plt.plot(results_df['threshold'], results_df['f1'], 'r-', label='F1-Score')
        plt.plot(results_df['threshold'], results_df['accuracy'], 'y-', label='Accuracy')
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title(f'Pengaruh Threshold pada Metrik Evaluasi (Query: {test_query})')
        plt.legend()
        plt.grid(True)
        plt.show()