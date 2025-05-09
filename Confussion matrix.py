import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay


dokumen = [
    "Carikan saya skripsi sistem pendukung keputusan",
    "mahasiswa lama melakukan kegiatan kampus",
    "penerimaan mahasiswa universitas nusantara",
    "orientasi kampus untuk mahasiswa baru telah selesai",
    "fasilitas kampus ditingkatkan oleh universitas",
    "jadwal kuliah semester ganjil telah diterbitkan",
    "pengumuman beasiswa untuk mahasiswa aktif",
    "universitas menyediakan layanan bimbingan akademik",
    "informasi pembayaran UKT tersedia di portal mahasiswa",
    "calon mahasiswa dapat mendaftar secara online"
]


y_pred = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


y_true = [0, 0, 0, 1, 0, 0, 1, 0, 0, 1]


df = pd.DataFrame({
    "Dokumen": dokumen,
    "Prediksi": y_pred,
    "Aktual": y_true,
    "Evaluasi": ["Benar" if y_pred[i] == y_true[i] else "Salah" for i in range(len(y_true))]
})


akurasi = accuracy_score(y_true, y_pred)
presisi = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)


print(df)
print("\nConfusion Matrix:")
print(cm)
print(f"\nAkurasi: {akurasi:.2f}")
print(f"Presisi: {presisi:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")


disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tidak Relevan", "Relevan"])
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
