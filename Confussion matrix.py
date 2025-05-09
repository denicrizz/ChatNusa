import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay

# 10 query uji coba
query = [
    "Carikan saya skripsi sistem pendukung keputusan",
    "skripsi pengelasan",
    "cari skripsi pembelajaran",
    "carikan skripsi metode bahan ajar",
    "cari skripsi mesin",
    "kapan kita bayar spp?",
    "pkkmb itu apa?",
    "dimana kita bayar spp?",
    "lokasi kampus unp kediri",
    "nama dosen teknik informatika"
]

# Label prediksi berdasarkan hasil sistem (1=relevan, 0=tidak)
y_pred = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# Label aktual (manual berdasarkan pemahaman)
y_true = [0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

# Buat tabel evaluasi
df = pd.DataFrame({
    "Query": query,
    "Prediksi": y_pred,
    "Aktual": y_true,
    "Evaluasi": ["Benar" if y_pred[i] == y_true[i] else "Salah" for i in range(len(y_true))]
})

# Hitung metrik evaluasi
akurasi = accuracy_score(y_true, y_pred)
presisi = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

# Tampilkan hasil tabel
print(df)
print("\nConfusion Matrix:")
print(cm)
print(f"\nAkurasi: {akurasi:.2f}  --> (TP + TN) / (TP + TN + FP + FN)")
print(f"Presisi: {presisi:.2f}  --> TP / (TP + FP)")
print(f"Recall : {recall:.2f}   --> TP / (TP + FN)")
print(f"F1-Score: {f1:.2f}      --> 2 * (Presisi * Recall) / (Presisi + Recall)")


# Tampilkan grafik confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Relevan", "Relevan"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
