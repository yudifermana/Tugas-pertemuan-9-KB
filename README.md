|       |                                   |
| ----- | --------------------------------- |
| Nama  | Yudi Fermana                      |
| Nim   | 312210321                         |
| Kelas | TI.22.A.SE.1                      |
| Tugas | Kecerdasan Buatan             |

# PENDAHULUAN

Dalam pengembangan sistem cerdas, akurasi model tunggal seringkali terbatas oleh bias atau varians data. Metode Ensemble hadir sebagai solusi dengan menggabungkan beberapa model pembelajaran mesin untuk meningkatkan ketahanan dan generalisasi model. Tugas ini bertujuan untuk mengimplementasikan teknik Ensemble Learning pada dataset sensor gas QCM (Quartz Crystal Microbalance) untuk mengklasifikasikan lima jenis alkohol (1-Octanol, 1-Propanol, 2-Butanol, 2-propanol, dan 1-isobutanol).
Tujuan

1. Mengimplementasikan metode Bagging menggunakan algoritma Random
   Forest.
2. Mengimplementasikan metode Boosting menggunakan algoritma Gradient
   Boosting.
3. Mengimplementasikan metode Max Voting untuk menggabungkan prediksi. 4. Mengevaluasi performa model berdasarkan metrik akurasi dan visualisasi confusion matrix.
   Metode Ensemble adalah teknik yang menggabungkan beberapa model untuk menyelesaikan masalah komputasi. Terdapat tiga pendekatan utama yang digunakan dalam tugas ini:
4. Bagging (Bootstrap Aggregating): Metode ini membangun beberapa model secara independen dan menggabungkan hasilnya. Algoritma yang digunakan adalah Random Forest.
5. Boosting: Metode ini bekerja secara sekuensial di mana model baru memperbaiki kesalahan model sebelumnya. Algoritma yang digunakan adalah Gradient Boosting.
6. Max Voting: Metode ini mengambil kelas dengan suara terbanyak dari
   beberapa model sebagai prediksi akhir.

# Persiapan Data (Data Preparation)

Dataset terdiri dari 5 file CSV yang digabungkan menjadi satu. Proses ini meliputi pemuatan data, penggabungan, dan konversi target dari One-Hot Encoding menjadi label tunggal.

## 1. Kode Persiapan Data

```py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# ================================================================
# 1. LOAD & PREPARE DATA
# ================================================================
def load_data():
    # List file sesuai struktur folder yang diupload
    file_paths = [
        'QCM3.csv',
        'QCM6.csv',
        'QCM7.csv',
        'QCM10.csv',
        'QCM12.csv'
    ]

    dfs = []
    for path in file_paths:
        # File menggunakan delimiter ';'
        df = pd.read_csv(path, sep=';')
        dfs.append(df)

    # Gabungkan seluruh file menjadi satu DataFrame
    combined = pd.concat(dfs, ignore_index=True)
    return combined


# Load Data
data = load_data()

# Pisahkan Fitur (X) dan Target (y)
X = data.iloc[:, :10]   # 10 sensor
y_raw = data.iloc[:, 10:]   # 5 kolom target One-Hot

# Konversi One-Hot menjadi 1 label (idxmax)
y = y_raw.idxmax(axis=1)

print("Total sampel data:", len(data))
print("Distribusi Kelas:\n", y.value_counts())

# Split Data (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

Keterangan: Potongan kode untuk memuat dataset QCM, menggabungkan file, dan membagi data latih/uji.

# Implementasi Model Ensemble

Berikut adalah implementasi tiga metode ensemble: Random Forest (Bagging), Gradient Boosting (Boosting), dan Voting Classifier.

## 2. Kode Implementasi Model

```py


# ================================================================
# 2. IMPLEMENTASI METODE ENSEMBLE
# ================================================================

# A. BAGGING → Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

# B. BOOSTING → Gradient Boosting
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
gb_model.fit(X_train, y_train)
y_pred_gb = gb_model.predict(X_test)

# C. VOTING (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='hard'
)
voting_clf.fit(X_train, y_train)
y_pred_vote = voting_clf.predict(X_test)


# ================================================================
# 3. EVALUASI
# ================================================================
print("\n============ HASIL EVALUASI ============\n")

print(f"Akurasi Random Forest (Bagging): {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Akurasi Gradient Boosting (Boosting): {accuracy_score(y_test, y_pred_gb):.4f}")
print(f"Akurasi Voting Classifier (Max Voting): {accuracy_score(y_test, y_pred_vote):.4f}")

print("\n--- Classification Report (Random Forest) ---")
print(classification_report(y_test, y_pred_rf))

# ================================================================
# 4. Confusion Matrix — Random Forest
# ================================================================
cm = confusion_matrix(y_test, y_pred_rf)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.xlabel("Prediksi")
plt.ylabel("Aktual")
plt.title("Confusion Matrix - Random Forest")
plt.show()
```

Keterangan: Implementasi algoritma Random Forest, Gradient Boosting, dan mekanisme Voting menggunakan pustaka Scikit-Learn.

# Hasil Evaluasi

Berdasarkan pengujian pada 20% data uji, diperoleh hasil akurasi dan visualisasi Confusion Matrix sebagai berikut:

## 3. Hasil Evaluasi dan Confusion Matrix

![gambar](foto/p10.1.png)
![gambar](foto/p10.png)
