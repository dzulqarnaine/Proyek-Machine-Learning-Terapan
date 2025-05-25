# Laporan Proyek Machine Learning - Muhammad Zukarnaini

## Domain Proyek

Menurut **[Lopana et al (2007)](https://www.tandfonline.com/doi/abs/10.1163/156939307783239429)**, diabetes adalah penyakit kronis yang menjadi masalah kesehatan besar di dunia. Penyakit ini ditandai dengan tingginya kadar glukosa dalam darah, yang disebabkan oleh ketidakmampuan tubuh dalam memproduksi insulin yang cukup atau disebabkan oleh gangguan dalam efektivitas insulin tersebut, atau keduanya. Diabetes terus meningkat secara global, mengancam kesehatan individu dan masyarakat. Jika tidak segera didiagnosis dan ditangani dengan tepat, diabetes dapat menimbulkan berbagai komplikasi serius, seperti gagal ginjal, penyakit jantung, hingga kerusakan saraf. Oleh karena itu, deteksi dini terhadap diabetes menjadi sangat penting, untuk memungkinkan intervensi lebih awal yang dapat mencegah terjadinya kerusakan lebih lanjut serta meningkatkan kualitas hidup penderita.

Penelitian oleh **[Digliati et al. (2017)](https://journals.sagepub.com/doi/full/10.1177/1932296817706375)**, menunjukkan bahwa pembelajaran mesin (_machine learning_) memiliki potensi besar dalam menganalisis data medis dan mengembangkan model prediktif untuk komplikasi diabetes tipe 2, khususnya yang berbasis data rekam medis elektronik. Dalam penelitian tersebut, digunakan pipeline data mining yang mencakup analisis profil klinis pasien, pembangunan model prediktif, dan validasi model. Penggunaan Random Forest untuk menangani data yang hilang dan regresi logistik dengan pemilihan fitur bertahap menghasilkan model yang efektif untuk memprediksi komplikasi diabetes seperti retinopati, neuropati, dan nefropati, dengan akurasi yang mencapai 0,838 dalam rentang waktu 3, 5, dan 7 tahun setelah kunjungan pertama pasien ke pusat diabetes. Faktor-faktor yang diperhitungkan dalam model meliputi jenis kelamin, usia, indeks massa tubuh (BMI), hemoglobin terglikasi (HbA1c), hipertensi, dan kebiasaan merokok, yang disesuaikan dengan jenis komplikasi dan periode waktu tertentu, memungkinkan model ini untuk diterapkan secara praktis dalam setting klinis.

Selain itu, **[Zou et al. (2018)](https://www.frontiersin.org/journals/genetics/articles/10.3389/fgene.2018.00515/full#B57)** mengungkapkan penerapan algoritma pembelajaran mesin seperti Decision Tree, Random Forest, dan Neural Network dalam memprediksi diabetes mellitus menggunakan data pemeriksaan fisik rumah sakit di Luzhou, China, dengan 14 atribut kesehatan. Untuk mengatasi masalah ketidakseimbangan data, dilakukan ekstraksi data secara acak sebanyak lima kali, dan hasilnya dirata-ratakan untuk meningkatkan kualitas prediksi. Selain itu, metode reduksi dimensi seperti Principal Component Analysis (PCA) dan Minimum Redundancy Maximum Relevance (mRMR) diterapkan untuk meningkatkan performa model. Hasil penelitian tersebut menunjukkan bahwa algoritma Random Forest memiliki akurasi tertinggi sebesar 0,8084 ketika semua atribut digunakan, menyoroti keunggulan pendekatan ini dalam memprediksi diabetes berdasarkan data kesehatan yang tersedia.

Melihat temuan-temuan ini, penggunaan pembelajaran mesin dalam prediksi diabetes menjadi solusi yang sangat menjanjikan untuk meningkatkan efektivitas deteksi dini penyakit ini. Dengan model berbasis machine learning, kita dapat mengidentifikasi individu berisiko tinggi secara lebih cepat dan akurat. Hal ini memungkinkan tindakan pencegahan yang lebih tepat waktu dan keputusan medis yang lebih baik terkait gaya hidup dan perawatan. Prediksi berbasis data ini tidak hanya membantu tenaga medis dalam membuat diagnosis lebih cepat, tetapi juga memberikan rekomendasi yang lebih akurat tentang pengelolaan diabetes dan pencegahan komplikasi jangka panjang.

Selanjutnya, implementasi teknologi ini dapat mempercepat perubahan positif dalam sistem kesehatan global. Dengan model prediksi yang dapat diandalkan, kita tidak hanya mengurangi risiko komplikasi jangka panjang, tetapi juga menurunkan biaya pengobatan yang mahal. Masyarakat pun dapat lebih sadar akan faktor risiko yang terkait dengan diabetes dan didorong untuk mengadopsi gaya hidup yang lebih sehat. Mengingat perkembangan pesat dalam kecerdasan buatan (AI) dan pembelajaran mesin, masa depan prediksi medis semakin cerah. Teknologi ini dapat disempurnakan untuk mencakup lebih banyak variabel dan faktor yang lebih mendalam dalam analisis, yang pada akhirnya dapat menghasilkan solusi lebih personal dan lebih efektif dalam perawatan kesehatan. Dengan demikian, model prediksi diabetes berbasis machine learning berpotensi untuk mengubah paradigma perawatan kesehatan global dan memberikan manfaat besar bagi individu, masyarakat, serta sistem kesehatan secara keseluruhan.

## Business Understanding

### Problem Statements

1. Bagaimana cara memprediksi kemungkinan seseorang mengidap diabetes berdasarkan berbagai faktor kesehatan yang dimilikinya?
2. Sejauh mana akurasi model dalam memprediksi diabetes jika dibandingkan dengan metode tradisional dalam deteksi penyakit ini?

### Goals

1. Membangun sebuah model pembelajaran mesin yang dapat memprediksi kemungkinan seseorang mengidap diabetes berdasarkan faktor-faktor kesehatan yang ada.
2. Mengevaluasi kinerja model pembelajaran mesin dengan menggunakan berbagai metrik evaluasi seperti akurasi, presisi, recall, F1-score, dan confusion matrix, untuk memastikan bahwa model dapat mendeteksi diabetes secara efektif dan optimal.

### Solution Statement

1. Menerapkan beberapa algoritma pembelajaran mesin, seperti Random Forest, Decision Tree, dan Naive Bayes, untuk membandingkan kinerja model dalam memprediksi diabetes.
2. Menganalisis hasil dari masing-masing model dengan menggunakan metrik evaluasi yang telah disebutkan untuk memilih model yang paling efisien dan akurat dalam memberikan prediksi tentang kemungkinan seseorang mengidap diabetes.

## Data Understanding

**`Diabetes prediction dataset`** adalah kumpulan data yang mencakup informasi medis dan demografi dari pasien beserta status diabetes mereka (positif atau negatif). Dataset ini mencakup berbagai fitur, seperti usia, jenis kelamin, indeks massa tubuh (BMI), hipertensi, penyakit jantung, riwayat merokok, kadar HbA1c, dan kadar glukosa darah. Terdapat 19 kolom dan 100.000 baris dalam dataset ini, yang sudah dibersihkan dan tidak mengandung nilai yang hilang. Dataset ini diambil dari platform **[Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset)**.

### Variabel-variabel pada Diabetes prediction dataset adalah sebagai berikut:

- **`gender`** :Jenis kelamin yang merujuk pada perbedaan biologis antara laki-laki dan perempuan, yang dapat memengaruhi risiko seseorang terhadap diabetes. Terdapat tiga kategori pada variabel ini: laki-laki, perempuan, dan lainnya.
- **`age`** : Usia merupakan faktor yang sangat penting karena diabetes lebih sering didiagnosis pada individu yang lebih tua. Dalam dataset ini, rentang usia yang tercatat adalah antara 0 hingga 80 tahun.
- **`hypertension`** : Hipertensi adalah kondisi medis di mana tekanan darah tinggi dapat meningkatkan risiko penyakit, termasuk diabetes. Nilainya adalah 0 atau 1, di mana 0 berarti tidak memiliki hipertensi dan 1 berarti menderita hipertensi.
- **`heart_disease`** : Penyakit jantung merupakan faktor risiko lain yang berkontribusi pada peningkatan kemungkinan seseorang mengidap diabetes. Nilai pada variabel ini adalah 0 atau 1, dengan 0 menunjukkan tidak ada penyakit jantung dan 1 berarti pasien memiliki penyakit jantung.
- **`smoking_history`** : Riwayat merokok berperan sebagai faktor tambahan dalam peningkatan risiko diabetes, serta memperburuk dampak komplikasi. Terdapat 5 kategori yang tersedia dalam data: tidak merokok saat ini, merokok sebelumnya, tidak ada informasi, merokok saat ini, dan merokok pernah.
- **`bmi`** : Indeks Massa Tubuh (BMI) mengukur komposisi tubuh berdasarkan berat dan tinggi badan. BMI yang tinggi sering dikaitkan dengan peningkatan risiko diabetes. Rentang BMI dalam dataset ini adalah dari 10,16 hingga 71,55. Kategori BMI adalah: kurang dari 18,5 (kekurangan berat badan), 18,5 hingga 24,9 (normal), 25 hingga 29,9 (kelebihan berat badan), dan 30 atau lebih (obesitas).
- **`HbA1c_level`** : Kadar HbA1c (Hemoglobin A1c) mengukur rata-rata kadar gula darah dalam dua hingga tiga bulan terakhir. Kadar HbA1c yang lebih tinggi menunjukkan risiko lebih besar untuk diabetes. Biasanya, kadar lebih dari 6,5% menandakan adanya diabetes.
- **`blood_glucose_level`** : Kadar glukosa darah mengacu pada jumlah glukosa yang ada dalam darah pada suatu waktu tertentu. Kadar glukosa darah yang tinggi merupakan indikator utama diabetes.
- **`diabetes`** : Variabel target yang akan diprediksi, dengan nilai 1 menunjukkan pasien mengidap diabetes dan 0 berarti tidak mengidap diabetes.

### Visualisasi Distribusi Data Numerik

Visualisasi distribusi untuk variabel numerik menunjukkan beberapa pola yang menarik. Variabel **age** menunjukkan distribusi yang cukup normal, meskipun ada sedikit skew di ujung kanan, yang menunjukkan bahwa sebagian besar sampel terfokus pada usia yang lebih muda, namun ada sebagian kecil individu dengan usia yang lebih tua. Sementara itu, **BMI** memiliki distribusi yang sangat tidak merata dengan puncak yang tajam di sekitar nilai rendah, yang mengindikasikan banyak individu dengan **BMI** rendah, tetapi sebagian kecil memiliki nilai **BMI** sangat tinggi. **Blood_glucose_level** memperlihatkan distribusi multimodal, dengan beberapa puncak yang menunjukkan adanya subkelompok dengan kadar glukosa darah yang berbeda-beda, mencerminkan variasi dalam status kesehatan individu. **HbA1c_level** juga menunjukkan pola distribusi yang serupa, dengan banyak puncak yang menandakan adanya kelompok individu dengan berbagai tingkat kadar HbA1c, dan distribusi yang tidak merata.

![Numerik](./assets/images/Distribusi_Numerik.png)

### Visualisasi Distribusi Data Kategori

Visualisasi distribusi variabel kategori mengungkapkan beberapa pola yang berbeda. **Gender** menunjukkan bahwa jumlah individu perempuan lebih banyak dibandingkan laki-laki dalam dataset ini. Untuk variabel **hypertension**, mayoritas individu tidak memiliki hipertensi, dengan hanya sebagian kecil yang menderita hipertensi. Hal serupa juga berlaku untuk **heart_disease** dan **diabetes**, di mana sebagian besar individu tidak memiliki penyakit jantung atau diabetes, dengan hanya sejumlah kecil yang terdiagnosis. Pada kategori **smoking_history**, terlihat bahwa sebagian besar data termasuk dalam kategori "No Info" dan "Never", sementara kategori seperti "Former", "Current", "Not Current", dan "Ever" memiliki jumlah yang jauh lebih sedikit. Ini mungkin mencerminkan kebiasaan merokok yang tidak terdokumentasi dengan baik dalam dataset atau menunjukkan kurangnya riwayat merokok di kalangan sebagian besar individu.

![Kategori](./assets/images/Distribusi_Kategori.png)

### Visualisasi Rata Rata Diabetes Dibandingkan Fitur Lain

Visualisasi yang menunjukkan perbandingan rata-rata diabetes berdasarkan fitur-fitur tertentu memberikan wawasan penting. Dapat dilihat bahwa **laki-laki** memiliki tingkat rata-rata diabetes yang lebih tinggi dibandingkan perempuan, menunjukkan perbedaan dalam prevalensi diabetes berdasarkan jenis kelamin. Selain itu, individu dengan **hipertensi** dan **penyakit jantung** menunjukkan prevalensi diabetes yang lebih tinggi, mencerminkan hubungan yang kuat antara kondisi medis ini dan peningkatan risiko diabetes. Untuk **riwayat merokok**, individu yang memiliki riwayat merokok sebelumnya (Former) menunjukkan rata-rata diabetes tertinggi, sementara individu yang memiliki kategori No Info cenderung memiliki tingkat rata-rata diabetes yang lebih rendah, mungkin karena data yang kurang lengkap atau kurang terdeteksi.

![Mean](./assets/images/MeanDiabetesVSOtherFeatures.png)

### Visualisasi Kernel Density Estimation

Visualisasi KDE (Kernel Density Estimation) memperlihatkan hubungan antara berbagai fitur dalam dataset. Scatter plot menunjukkan distribusi titik data di antara pasangan variabel yang relevan, memberikan gambaran visual mengenai korelasi antar fitur. Sedangkan plot KDE yang terletak di diagonal menggambarkan distribusi probabilitas dari setiap variabel. Beberapa fitur seperti HbA1c_level dan blood_glucose_level menunjukkan distribusi yang lebih bervariasi, yang mengindikasikan adanya berbagai level keparahan kondisi diabetes di antara individu. Sementara itu, variabel biner seperti hypertension dan heart_disease menunjukkan titik data yang lebih terpisah, dengan sedikit pola dalam scatter plot, yang mengindikasikan bahwa faktor-faktor ini mungkin tidak memiliki variasi yang cukup untuk memberikan gambaran yang jelas tentang hubungan mereka dengan diabetes.

![Mean](./assets/images/KDE.png)

### Visualisasi Correlation Matrix

Matriks korelasi memberikan wawasan tentang hubungan antar fitur numerik dalam dataset. Dari matriks ini, dapat terlihat bahwa HbA1c_level dan blood_glucose_level memiliki korelasi positif yang moderat dengan diabetes (0.38 dan 0.39), yang menunjukkan bahwa kadar gula darah dan HbA1c berperan penting dalam menentukan status diabetes. Korelasi yang cukup tinggi ini memperlihatkan bahwa kadar glukosa darah yang tinggi berhubungan langsung dengan kemungkinan seseorang mengidap diabetes. Selain itu, BMI juga memiliki korelasi moderat dengan age (0.38), yang menunjukkan bahwa seiring bertambahnya usia, banyak individu cenderung mengalami peningkatan berat badan, yang pada gilirannya dapat meningkatkan risiko diabetes. Namun, faktor seperti heart_disease dan hypertension menunjukkan korelasi yang lebih rendah dengan diabetes, yang menunjukkan bahwa meskipun kedua faktor ini dapat berkontribusi pada risiko diabetes, mereka mungkin tidak sebesar peran faktor lain seperti kadar glukosa darah atau HbA1c dalam menentukan kemungkinan diabetes.

![Mean](./assets/images/Matriks-Korelasi.png)

## Data Preparation

- **`Category Handling`** : Dalam fitur gender, terdapat kategori langka yang hanya muncul 18 kali dari 100.000 baris data, yaitu kategori selain laki-laki dan perempuan. Oleh karena itu, kategori langka ini diganti dengan kategori mayoritas, yaitu "male" atau "female". Penggantian kategori langka seperti ini dengan modus atau kategori mayoritas membantu mengurangi noise dalam data serta mengatasi ketidakseimbangan yang mungkin terjadi. Dengan cara ini, model dapat belajar dari data yang lebih representatif, mengurangi kemungkinan distorsi yang disebabkan oleh kategori yang sangat jarang muncul, dan meningkatkan kualitas prediksi.

- **`Handling Outlier`** : Outlier pada fitur numerik diidentifikasi menggunakan metode Interquartile Range (IQR). Setelah outlier ditemukan, dilakukan teknik clipping untuk membatasi nilai-nilai yang berada di luar rentang batas yang telah ditentukan. Outlier bisa memberikan pengaruh yang tidak proporsional dalam pelatihan model, sehingga clipping diterapkan untuk mengurangi efek ekstrem dan memastikan bahwa model dilatih menggunakan data yang lebih relevan dan representatif, tanpa terdistorsi oleh nilai-nilai yang sangat jauh dari distribusi data utama.

  ```python
    numeric = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    outlierValues = {}
    data_before_clipping = data[numeric].copy()

    # Deteksi dan Clipping Outlier
    for col in numeric:
        q1, q3 = np.percentile(data[col].dropna(), [25, 75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr

        outliers = data[col][(data[col] < lower) | (data[col] > upper)]
        outlierValues[col] = outliers

        data[col] = np.clip(data[col], lower, upper)
  ```

- **`Encoding Fitur Kategori`** : Fitur-fitur kategori seperti gender dan smoking_history diencoding menjadi format numerik menggunakan OneHotEncoder. Pengkodean ini penting karena algoritma pembelajaran mesin membutuhkan data dalam format numerik untuk dapat memproses dan mempelajari pola-pola yang ada. Dengan melakukan pengkodean, informasi dalam data kategori tetap dipertahankan, sementara data yang awalnya bersifat kategorikal diubah menjadi format yang sesuai untuk dimasukkan ke dalam model pembelajaran mesin.
  ```python
    categoric = ['gender', 'smoking_history']
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_data = pd.DataFrame(
        encoder.fit_transform(data[categoric]),
        columns=encoder.get_feature_names_out(categoric),
        index=data.index)
    data = data.drop(columns=categoric).join(encoded_data)
  ```
- **`Standarisasi`** : Untuk fitur numerik seperti age dan bmi, dilakukan standarisasi menggunakan StandardScaler. Proses ini memastikan bahwa nilai-nilai fitur berada dalam rentang yang lebih seragam, dengan mayoritas nilai berada dalam rentang -3 hingga 3. Standarisasi ini penting karena perbedaan skala antar fitur bisa mempengaruhi kinerja model, terutama pada algoritma pembelajaran mesin yang sensitif terhadap skala data. Dengan menstandarisasi data, kita membantu model agar lebih cepat dan stabil dalam proses optimasi, serta menghindari bias pada fitur yang memiliki rentang nilai jauh lebih besar.
  ```python
    numeric = ['age', 'bmi', 'blood_glucose_level', 'HbA1c_level']
    scaler = StandardScaler()
    data[numeric] = scaler.fit_transform(data[numeric])
  ```
- **`Spliting Data`** : Dataset dibagi menjadi dua bagian, yaitu 80% untuk data pelatihan dan 20% untuk data pengujian. Target atau label yang diprediksi adalah kolom diabetes. Pemisahan data ini sangat penting untuk mengevaluasi kinerja model pada data yang tidak dilihat selama pelatihan. Dengan cara ini, kita dapat mengukur kemampuan model untuk menggeneralisasi dan menghindari overfitting. Pemisahan data juga memungkinkan kita untuk menguji model di lingkungan dunia nyata dengan data baru yang belum pernah dipelajari sebelumnya.
  ```python
    X = data.drop(columns=["diabetes"])
    y = data["diabetes"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  ```

## Modeling

Pada Projek ini, model yang digunakan untuk memprediksi kemungkinan seseorang mengidap diabetes berdasarkan fitur-fitur yang tersedia adalah Decision Tree, Random Forest, dan Naive Bayes. Pemilihan ketiga model ini didasarkan pada karakteristik dan keunggulannya masing-masing dalam menangani masalah prediksi penyakit diabetes. Berikut adalah alasan pemilihan model-model tersebut:

- **Decision Tree**: Decision Tree adalah model yang mudah dipahami dan sangat efektif dalam klasifikasi. Keunggulannya adalah kemampuan untuk menangani data non-linear dengan baik dan memberikan interpretasi yang jelas mengenai proses pengambilan keputusan. Meskipun demikian, model ini cenderung rentan terhadap overfitting, terutama jika pohon keputusan terlalu dalam. Oleh karena itu, teknik pruning atau pembatasan kedalaman pohon sering digunakan untuk mengatasi masalah ini.
- **Random Forest**: Ini adalah metode ensemble yang menggabungkan banyak pohon keputusan (decision tree) untuk menghasilkan keputusan yang lebih robust. Kelebihan utama dari Random Forest adalah kemampuannya dalam menangani data non-linear dan kemampuannya untuk mengurangi risiko overfitting dibandingkan dengan model pohon keputusan tunggal. Meskipun demikian, Random Forest bisa lebih lambat dalam proses pelatihan dan prediksi ketika dibandingkan dengan model boosting.
- **Naive Bayes**: Model ini berbasis pada teorema Bayes dengan asumsi independensi antar fitur. Keunggulan utama dari Naive Bayes adalah kemampuannya dalam memproses data dalam jumlah besar dengan cepat dan efisien. Meskipun asumsi independensi fitur terkadang tidak valid, Naive Bayes sering memberikan hasil yang baik untuk masalah klasifikasi sederhana dan berguna dalam kondisi data yang tidak terlalu rumit.

Tahapan yang Dilakukan dalam Proses Pemodelan:

1. **`Load Model`**:

   - **Random Forest** dimuat dengan parameter `n_estimators=100` dan `random_state=42`:
     ```python
     random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
     ```
   - **Decision Tree** dimuat dengan parameter `random_state=42`:
     ```python
     decision_tree = DecisionTreeClassifier(random_state=42)
     ```
   - **Naive Bayes** :
     ```python
     naive_bayes = BernoulliNB()
     ```

2. **`Pelatihan Model`**:

   - **Random Forest** dilatih menggunakan data pelatihanu `X_train dan y_train`:
     ```python
     random_forest.fit(X_train, y_train)
     ```
   - **Decision Tree** dilatih menggunakan data pelatihanu `X_train dan y_train`:
     ```python
     decision_tree.fit(X_train, y_train)
     ```
   - **Naive Bayes** dilatih menggunakan data pelatihanu `X_train dan y_train`:
     ```python
     naive_bayes.fit(X_train, y_train)
     ```

3. **Evaluasi Model**:
   Hasil pelatihan dari ketiga model dibandingkan untuk menentukan model terbaik berdasarkan metrik evaluasi.

Setelah melatih ketiga model, hasil prediksi dari masing-masing model dievaluasi menggunakan metrik yang relevan, seperti akurasi, precision, recall, dan F1-score. Hal ini dilakukan untuk membandingkan kinerja masing-masing model dan memilih model terbaik yang dapat memberikan prediksi yang lebih akurat.

Setelah evaluasi awal, Random Forest dipilih sebagai model terbaik berdasarkan kinerja prediksi yang paling optimal dibandingkan dengan Decision Tree dan Naive Bayes. Model ini memberikan hasil yang paling akurat dalam mendeteksi diabetes, meskipun perbandingan lebih lanjut menunjukkan bahwa parameter model harus lebih lanjut disesuaikan untuk meningkatkan kestabilan prediksi pada dataset yang lebih besar.

## Evaluation

**Evaluasi model** dilakukan dengan menggunakan sejumlah metrik kunci yang relevan untuk masalah klasifikasi biner, di antaranya **Accuracy**, **Precision**, **Recall**, **F1-Score**, dan **Confusion Matrix**. Metrik-metrik ini dipilih karena dataset yang digunakan berkaitan dengan prediksi kondisi kesehatan (kemungkinan seseorang mengidap diabetes), di mana penting untuk mempertimbangkan keseimbangan antara deteksi kasus positif dan negatif. Evaluasi ini bertujuan untuk memastikan bahwa model tidak hanya memprediksi dengan tepat, tetapi juga mengurangi kesalahan dalam mendeteksi pasien yang berisiko tinggi.

Metrik Evaluasi yang Digunakan

1. **`Accuracy Score`** :

   - **Accuracy**: Mengukur persentase prediksi yang benar dari total data. Metrik ini memberikan gambaran umum tentang kinerja model, meskipun dalam beberapa kasus, seperti ketidakseimbangan kelas, akurasi saja bisa menyesatkan.

$$
\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}
$$

```python
print(f"Akurasi: {accuracy_score(y_test, y_pred_dt):.4f}")
```

2. **`Classification Report`** :

   - **Precision**: Menunjukkan seberapa banyak dari prediksi positif yang benar-benar positif. Dalam konteks ini, precision menggambarkan seberapa banyak pasien yang diprediksi mengidap diabetes benar-benar mengidapnya. Precision yang tinggi mengurangi risiko memberikan diagnosis positif yang salah.

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

- **Recall (Sensitivity)**: Mengukur seberapa banyak kasus positif yang benar-benar berhasil dideteksi oleh model. Recall sangat penting dalam situasi medis, di mana kegagalan untuk mendeteksi seseorang yang mengidap diabetes (false negative) bisa berakibat fatal.

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

- **F1-Score**: Merupakan rata-rata harmonis antara precision dan recall. F1-score memberikan gambaran yang lebih seimbang antara keduanya, terutama jika data cenderung tidak seimbang, dan membantu menilai trade-off antara precision dan recall.

$$
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
$$

```python
print("\nClassification Report:\n", classification_report(y_test, y_pred))
```

3. **`Confusion Matrix`** : Matriks ini memberikan rincian dari prediksi model, mengklasifikasikan jumlah true positives, true negatives, false positives, dan false negatives. Hal ini sangat berguna untuk menganalisis jenis kesalahan yang dilakukan oleh model, misalnya, apakah lebih sering gagal mendeteksi diabetes atau mengidentifikasi orang sehat sebagai penderita diabetes.

   |                    | Predicted Negatif (0) | Predicted Positif (1) |
   | ------------------ | --------------------- | --------------------- |
   | Actual Negatif (0) | True Negative (TN)    | False Positive (FP)   |
   | Actual Positif (1) | False Negative (FN)   | True Positive (TP)    |

   ```python
   cm_dt = confusion_matrix(y_test, y_pred_dt)
   ```

Berikut adalah ringkasan hasil evaluasi berdasarkan prediksi pada data :

1. Accuracy dan Classification Report :

   | Model         | Accuracy | Precision | Recall | F1-Score |
   | ------------- | -------- | --------- | ------ | -------- |
   | Random Forest | 0.9705   | 0.94      | 0.69   | 0.80     |
   | Decision Tree | 0.9525   | 0.72      | 0.73   | 0.72     |
   | Naive Bayes   | 0.9705   | 0.94      | 0.68   | 0.80     |

   Analisis Hasil:

   - Accuracy : Ketiga model menunjukkan akurasinya sangat tinggi (sekitar 97% untuk Random Forest dan Naive Bayes, dan sekitar 95% untuk Decision Tree), yang menunjukkan bahwa semua model mampu memprediksi dengan sangat baik pada data uji. Meskipun Naive Bayes dan Random Forest memiliki akurasi yang sedikit lebih tinggi, perbedaannya sangat kecil.

   - Precision : Semua model memiliki precision tinggi (~0.94 - 0.96), yang berarti model jarang memberikan prediksi positif yang salah (False Positive). Ini menunjukkan bahwa model cukup andal dalam memprediksi orang yang benar-benar mengidap diabetes.

   - Recall : Nilai recall sedikit lebih rendah (~0.67 - 0.73), yang menunjukkan bahwa masih ada beberapa kasus positif yang tidak terdeteksi dengan baik oleh model (False Negative). Ini penting, karena False Negatives berarti ada orang yang mengidap diabetes tetapi tidak terdeteksi, yang bisa berisiko untuk kesehatan mereka.

   - F1-Score : memberikan keseimbangan antara precision dan recall, memberikan gambaran yang lebih lengkap. Decision Tree dan Naive Bayes menunjukkan nilai F1-Score yang hampir identik (~0.80), memberikan keseimbangan yang lebih baik antara precision dan recall dibandingkan dengan Random Forest (F1-Score sekitar 0.80).

   Berdasarkan hasil evaluasi, meskipun **`Naive Bayes`** dan **Random Forest** menunjukkan akurasi tertinggi (0.9705), **Decision Tree** dipilih sebagai model terbaik. Meskipun sedikit lebih rendah dalam hal akurasi dibandingkan dengan **`Naive Bayes`** dan **Random Forest**, **Decision Tree** menunjukkan keseimbangan yang lebih baik antara Precision dan Recall. Hal ini penting, karena model yang mampu mendeteksi lebih banyak kasus positif (dengan recall yang lebih tinggi) dan memberikan sedikit kesalahan positif (precision yang tinggi) lebih diutamakan dalam aplikasi medis untuk prediksi diabetes.

2. Analisis Berdasarkan Confusion Matrix:

   | Model         | Actual             | Predicted Negatif (0) | Predicted Positif (1) |
   | ------------- | ------------------ | --------------------- | --------------------- |
   | Random Forest | Actual Negatif (0) | 18213                 | 79                    |
   | Random Forest | Actual Positif (1) | 529                   | 1179                  |
   | Decision Tree | Actual Negatif (0) | 17805                 | 487                   |
   | Decision Tree | Actual Positif (1) | 464                   | 1244                  |
   | Naive Bayes   | Actual Negatif (0) | 17798                 | 494                   |
   | Naive Bayes   | Actual Positif (1) | 1179                  | 529                   |

   Berdasarkan hasil dari confusion matrix untuk ketiga model, berikut adalah poin-poin penting yang perlu diperhatikan:

   1. Performa dalam Mengklasifikasikan Kelas Negatif (0):
      Ketiga model menunjukkan performa yang sangat baik dalam mengklasifikasikan kelas negatif (0), yang tercermin dalam jumlah True Negative (TN) yang sangat tinggi dan False Positive (FP) yang sangat rendah. Hal ini menunjukkan bahwa model-model tersebut jarang salah mengklasifikasikan individu yang tidak mengidap diabetes sebagai positif. Sebagai contoh, Random Forest mengklasifikasikan 18213 kasus negatif dengan benar, sementara hanya 79 yang salah diklasifikasikan sebagai positif.

   2. Performa dalam Mengklasifikasikan Kelas Positif (1):
      Meskipun ketiga model sangat baik dalam mendeteksi kelas negatif, ada perbedaan signifikan dalam cara mereka menangani kelas positif (1).

   - Naive Bayes menunjukkan jumlah False Negative (FN) tertinggi, dengan 1179 kasus positif yang tidak terdeteksi dengan baik (dikelompokkan sebagai negatif). Ini berarti model Naive Bayes lebih sering gagal mendeteksi individu yang benar-benar mengidap diabetes.
   - Decision Tree memiliki False Negative yang lebih rendah (464) dibandingkan Naive Bayes dan False Positive yang lebih rendah dibandingkan dengan Random Forest (529). Dengan demikian, Decision Tree lebih baik dalam mendeteksi kasus positif tanpa terlalu banyak menghasilkan kesalahan klasifikasi.

   3. Keunggulan Decision Tree:
      Secara keseluruhan, Decision Tree menunjukkan keseimbangan terbaik dalam menangani kedua kelas. Model ini memiliki False Negative lebih rendah dibandingkan Naive Bayes dan False Positive lebih rendah dibandingkan Random Forest. Hal ini menjadikannya model yang lebih optimal untuk mendeteksi kasus positif (diabetes) dengan lebih akurat dan dengan lebih sedikit kesalahan klasifikasi, yang sangat penting dalam konteks medis di mana deteksi dini sangat krusial.
