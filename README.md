# Laporan Proyek Machine Learning - Parveen Uzma Habidin
## Domain Proyek
Domain yang dipilih untuk proyek *machine learning* ini adalah **Pertanian**, dengan judul **Predictive Analytics: Kualitas Pisang**  

### Latar Belakang

![Pisang](https://i0.wp.com/blog.alfagift.id/wp-content/uploads/2024/05/jenis-pisang.jpg)


Indonesia merupakan salah satu negara penghasil pisang terbesar di dunia, dengan produksi mencapai lebih dari 9,34 juta ton per tahun. Pisang menjadi komoditas penting bagi petani dan berkontribusi pada perekonomian nasional.[[1](https://www.bps.go.id/id/statistics-table/2/NjIjMg==/produksi-tanaman-buah-buahan.html)] Salah satu tantangan utama dalam industri pisang adalah menjaga kualitas produk. Kualitas pisang dapat menurun akibat berbagai faktor, seperti tingkat kematangan, warna kulit, kerusakan mekanis, serta kondisi penyimpanan dan distribusi. Penurunan kualitas pisang dapat menyebabkan kerugian ekonomi bagi petani, pedagang, dan distributor.[[2](https://doi.org/10.12928/jstie.v10i2.22390)]

Penerapan _predictive analytics_ dalam industri pisang dapat memberikan manfaat bagi petani, distributor, dan konsumen. Petani dapat meningkatkan keuntungan dengan mengoptimalkan waktu panen dan memperkirakan kualitas pisang secara otomatis. Distributor dapat mengurangi kerugian akibat kerusakan pascapanen dan meningkatkan efisiensi logistik. Konsumen pun mendapatkan produk dengan kualitas yang lebih baik, umur simpan lebih panjang, dan harga yang lebih stabil.[[3](https://repository.bsi.ac.id/repo/files/440819/download/Jurnal_IJCIT_Ahmad-Rifqi.pdf)]

## Business Understanding
Salah satu tantangan utama dalam industri pisang adalah menjaga kualitas produk setelah panen. Penurunan kualitas akibat kematangan berlebih, kerusakan fisik, atau penyimpanan yang tidak sesuai dapat menyebabkan kerugian ekonomi bagi petani dan distributor. Untuk mengatasi masalah ini, diperlukan model prediksi kualitas pisang berbasis machine learning yang dapat membantu mengklasifikasikan mutu buah secara otomatis. Dengan sistem ini, petani dapat menentukan waktu panen yang optimal, distributor dapat melakukan sortasi lebih efisien, dan konsumen menerima produk dengan kualitas yang lebih baik.

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini merupakan rincian masalah yang dapat diselesaikan pada proyek ini:
- Bagaimana membangun model machine learning yang mampu mengklasifikasikan kualitas pisang berdasarkan data visual?
- Algoritma machine learning apa yang menghasilkan akurasi prediksi kualitas pisang paling optimal?
- Bagaimana model prediktif ini dapat membantu petani dan distributor dalam mengurangi kerugian akibat kerusakan dan meningkatkan nilai jual produk?

### Goals
Tujuan dari proyek ini adalah:
- Merancang model machine learning yang mampu mengklasifikasikan kualitas pisang dengan mengandalkan data fitur numerik maupun visual.
- Melakukan evaluasi terhadap berbagai algoritma machine learning untuk mengidentifikasi model dengan performa terbaik dalam prediksi kualitas pisang.
- Membangun solusi berbasis data yang dapat dimanfaatkan oleh petani dan distributor untuk meningkatkan efisiensi dalam proses sortasi dan distribusi pisang.

### Solution Statements
- Menganalisis data dengan melakukan univariate dan multivariate analysis. Pemahaman terhadap distribusi data juga didukung dengan visualisasi untuk mengetahui korelasi antar fitur dan mendeteksi outlier.
- Melakukan proses data cleaning dan normalisasi agar data siap digunakan oleh algoritma machine learning untuk menghasilkan prediksi yang optimal.
- Membangun dan membandingkan beberapa variasi model machine learning untuk menentukan model terbaik dalam memprediksi kualitas pisang. Model-model yang digunakan antara lain:
    * K-Nearest Neighbor (KNN) adalah algoritma sederhana yang mengklasifikasikan data baru berdasarkan kesamaan dengan data terdekat. Algoritma ini sering digunakan sebagai baseline dalam klasifikasi karena kemudahan implementasi dan interpretasi.[[4]()]
    * Random Forest merupakan algoritma ensemble yang terdiri dari banyak decision tree. Setiap pohon memberikan prediksi, dan hasil akhir didapat dari mayoritas voting. Metode ini efektif dalam meningkatkan akurasi dan mengurangi risiko overfitting.[[5]()]
    * Support Vector Machine (SVM) bekerja dengan menemukan hyperplane terbaik yang memisahkan dua kelas dalam ruang berdimensi tinggi. SVM efektif untuk kasus klasifikasi biner dengan margin antar kelas yang jelas.[[6]()]
    * Naive Bayes adalah algoritma klasifikasi probabilistik berbasis teorema Bayes. Meskipun sederhana, algoritma ini sering memberikan performa kompetitif pada dataset dengan distribusi normal.[[7]()]

Extra Trees Classifier adalah algoritma berbasis ensemble yang mirip dengan Random Forest, namun melakukan pemilihan split secara lebih acak. Algoritma ini dapat mempercepat proses pelatihan dan sering memberikan akurasi tinggi.
[8]

## Data Understanding
### EDA - Deskripsi Variabel
**Informasi Datasets**


| Jenis | Keterangan |
| ------ | ------ |
| Title | _Banana Quality_ |
| Source | [Kaggle](https://www.kaggle.com/datasets/l3llff/banana) |
| Maintainer | [l3LlFF](https://www.kaggle.com/l3llff) |
| License | Apache 2.0 |
| Visibility | Publik |
| Tags | _Earth and Nature, Education, Food, Data Visualization, Exploratory Data Analysis, Binary Classification_ |
| Usability | 10.00 |

Berikut informasi pada dataset: 
Data yang digunakan dalam pembuatan model merupakan data primer yang disediakan secara publik di kaggle dengan nama datasets yaitu: _Banana Quality_

| Size | Weight | Sweetness | Softness | HarvestTime | Ripeness | Acidity | Quality |
| ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| -1.9249682 | 0.46807805 | 3.0778325 |-1.4721768 | 0.2947986 | 2.4355695	| 0.27129033  | good |
| -2.4097514 | 0.48686993 | 0.34692144 | -2.4950993 | -0.8922133 | 2.0675488 | 0.30732512  | good |
| -0.3576066 | 1.4831762 | 1.5684522 | -2.6451454 | -0.64726734 | 3.0906434	| 1.427322 | good |
| -0.8685235 | 1.5662014 | 1.8896049 | -1.2737614 | -1.0062776 | 1.8730015	| 0.47786173  | good |
| 0.65182525 | 1.3191992 | -0.022458995 | -1.2097088 | -1.430692 | 1.0783454	| 2.8124418  | good |


Tabel 1. EDA Deskripsi Variabel

Dilihat dari _Tabel 1. EDA Deskripsi Variabel_ dataset ini telah di *bersihkan* dan *normalisasi* terlebih dahulu oleh pembuat, sehingga mudah digunakan dan ramah bagi pemula. 
- Dataset berupa CSV (Comma-Seperated Values).
- Dataset memiliki 4001 sample dengan 9 fitur.
- Dataset memiliki 7 fitur bertipe float64 dan 2 fitur bertipe object.
- Terdapat 1 missing value dalam dataset.
### Variable - variable pada dataset
- `A_id` : Identifikasi unik untuk setiap buah.
- `Size` : Ukuran buah.
- `Weight` : Berat buah.
- `Sweetness` : Tingkat kemanisan buah.
- `Crunchiness` : Tekstur yang menunjukkan kerenyahan buah.
- `Juiciness` : Tingkat kesegaran buah.
- `Ripeness` : Tahap kematangan buah.
- `Acidity` : Tingkat keasaman buah.
- `Quality` : Kualitas buah secara keseluruhan, baik atau buruk.

Dari ke 9 fitur dapat dilihat bahwa fitur `A_id` tidak mempengaruhi kualitas buah hingga akan di hapus.

### EDA - Univariate Analysis

![Analisis Univariat (Data Kategori)](https://i.ibb.co/0MRrJCC/jumlah-kualitas-datasets.png)

Gambar 1a. Analisis Univariat (Data Kategori) 

![Univariate Analysis](https://i.ibb.co/V2mQ2dK/EDA-Univariate.png)

Gambar 1b. Analisis Univariat (Data Numerik) 

 Berdasarkan _Gambar 1a_ , dapat dilihat bahwa distribusi data katagorik _Quality_ yang terdiri dari _good_ dan _bad_ kualitas apel, yang mana nilai data **bad** terdiri dari `1928` dan **good** terdiri dari `1862`, yang mana menunjukan perbandingan data yang tidak terlalu jauh. Pada _Gambar 1b,_ untuk data numerik memiliki karakteristik, yaitu:
  - Dilihat dari distribusi data numerik _Size_, ukuran rata-rata buah berkisar dari -2 sampai 2, dan memiliki nilai rata-rata _Mean_ adalah -0.51.
  - Rata-rata berat apel bernilai -0.99 dan nilai _max_ berat apel adalah 3.08.
  - Rata-rata tingkat kemanisan apel -0.48.
  - Tekstur kerenyahan apel berkisar dari 0 hingga 2 yang mana nilai ini menunjukan rata-rata apel itu renyah.
  - Tingkat kesegaran buah dan Kematangan buat berada pada nilai 0.50 dan 0.53.
  - Rata-rata tingkat keasaman buah bernilai 0.06.

 Nilai-nilai ini menunjukkan bahwa data  telah dinormalisasi dengan cara _z-score normalization_ . _z-score normalization_  mengubah data dengan cara:
 - Mengurangi rata-rata (mean) dari setiap data point.
 - Membagi hasil pengurangan tersebut dengan standar deviasi data.
 

Pada kasus ini, rata-rata (mean) data "Size" adalah -0.51 dan standar deviasi data "Size" tidak diketahui. Namun, dengan nilai minimum -2 dan maksimum 2, dapat diasumsikan bahwa data "Size" telah diubah skalanya sehingga memiliki mean 0 dan standar deviasi 1. Data numerik lainnya, seperti _"Weight", "Sweetness", "Crunchiness", "Juiciness", "Ripeness", dan "Acidity"_, juga telah dinormalisasi dengan cara yang sama.


 

### EDA - Multivariate Analysis

![Multivariate Analysis](https://i.ibb.co/yNHmpNZ/EDA-MULTIVARIATE.png)


Gambar 2a. Analisis Multivariat

![Multivariate Analysis](https://i.ibb.co/WBQ5gPy/Matrix-corelasi.png)


Gambar 2b. Analisis Matriks Korelasi

Pada _Gambar 2a. Analisis Multivariat_, dengan menggunakan fungsi _pairplot_ dari _library seaborn_, tampak terlihat relasi pasangan dalam dataset menunjukan pola acak. Pada pola sebaran data grafik pairplot, terterlihat bahwa _Size_ dan _Sweetness_ memiliki korelasi negatif menurun, yang mana semakin kecil ukuran buah rasa nya akan semakin manis.
Pada _Gambar 2b. Analisis Matriks Korelasi_, merupakan _Correlation Matrix_ menunjukkan hubungan antar fitur dalam nilai korelasi. Jika diamati, fitur _Juiciness_ memiliki skor korelasi yang cukup besar `0.24` dengan fitur target _Acidity_ .
## Data Preparation
Pada proses _Data Preparation_ dilakukan kegiatan seperti _Data Gathering_, _Data Assessing_, dan _Data Cleaning_. Pada proses Data Gathering, data diimpor sedemikian rupa agar bisa dibaca dengan baik menggunakan dataframe Pandas. Untuk proses Data Assessing, berikut adalah beberapa pengecekan yang dilakukan:
- Duplicate data (data yang serupa dengan data lainnya).
- Missing value (data atau informasi yang "hilang" atau tidak tersedia)
- Outlier (data yang menyimpang dari rata-rata sekumpulan data yang ada).

Pada proses _Data Cleaning_ yang dilakukan adalah seperti:
- Converting Column Type (Mengubah tipe suatu kolom).
- Train Test Split (membagi data menjadi data latih dan data uji).
- Normalization (mentransformasi data ke dalam skala yang seragam sehingga semua fitur atau atribut memiliki rentang nilai yang sebanding).

| A_id | Size | Weight | Sweetness | Crunchiness | Juiciness | Ripeness | Acidity | Quality |
| ------ | ------ |------ | ------ | ------ | ------ |------ | ------ |------ |
| NaN | NaN | NaN | NaN |NaN | NaN| NaN	| Created_by_Nidula_Elgiriyewithana  | NaN |


Tabel 2. Melihat data missing value

Pada proyek kasus ini tidak ditemukannya data duplikat, tetapi ditemukannya _missing value_. Adapaun metode yang digunakan untuk mengatasi hal ini adalah dengan menerapkan _Dropping_ yaitu menghapus data yang _missing_ digunakannya metode ini dikarenakan jumlah missing value hanya berjumlah `1`. Lihat _Tabel 2. Melihat data missing value_. Adapun untuk _outlier_ juga dilakukan dengan metode _dropping_ menggunakan metode IQR.  IQR dihitung dengan mengurangkan kuartil ketiga (Q3) dari kuartil pertama (Q1) sebagaimana rumus berikut.

$$IQR = Q_3 - Q_1$$

- Q1 adalah kuartil pertama 
- Q3 adalah kuartil ketiga.

Setelah menggunakan metode IQR untuk menghilangkan _outlier_ pada dataset jumlah dataset menjadi `3790` yang awalnya adalah `4000`.
Pada proyek ini digunakan _Train Test Split_ pada library  *sklearn.model_selection* untuk membagi dataset menjadi data latih dan data uji dengan pembagian sebesar 20:80 dan random state sebesar 60. Pada proyek kasus ini digunakan _Normalization_ pada library _sklearn.preprocessing.MinMaxScaler_ untuk menormalisasi dataset. Semua proses ini diperlukan dalam rangka membuat model yang baik.
## Modeling
Algoritma pada proyek ini melakukan pemodelan dengan 5 algoritma, yaitu:

 _K-Nearest Neighbors (KNN)_ adalah algoritma machine learning yang sederhana dan mudah dipahami untuk klasifikasi dan regresi. Algoritma ini bekerja dengan menemukan k tetangga terdekat dari data baru dan kemudian menggunakan kategori atau nilai rata-rata dari tetangga tersebut untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
-  `n_neighbors` jumlah tetangga terdekat.
- `weight = distance` Tetangga yang lebih dekat memiliki pengaruh lebih besar.

Keunggulan _KNN_ :
- Dapat digunakan untuk klasifikasi dan regresi.
- Sederhana dan mudah dipahami.

Kerugian _KNN_ :
- Sensitif terhadap outlier. 
- Membutuhkan banyak memori dan waktu komputasi untuk dataset besar. 
- Sulit untuk memilih nilai K yang optimal.

 _Random Forest_ adalah algoritma machine learning ensemble yang menggabungkan beberapa decision tree untuk meningkatkan akurasi prediksi. Algoritma ini bekerja dengan membuat banyak decision tree secara acak dan kemudian menggunakan voting untuk memprediksi kategori atau nilai data baru. Adapun parameter yang digunakan pada proyek ini adalah:
- `max_depth` kedalaman maksimum.

Keunggulan _Random Forest_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.

Kerugian _Random Forest_ :
- Cenderung overfit pada dataset kecil. 
- Membutuhkan banyak waktu komputasi untuk pelatihan. 
- Sulit untuk diinterpretasikan.

 _Support Vector Machine (SVM)_ adalah algoritma machine learning yang digunakan untuk klasifikasi dan regresi. Algoritma ini bekerja dengan mencari hyperplane yang memisahkan data menjadi dua kelas dengan margin terbesar. Parameter yang digunakan pada SVM kali ini adalah parameter bawaan.
 
 Keuntungan  _Support Vector Machine (SVM)_ :
- Memiliki akurasi prediksi yang tinggi.
- Mampu menangani dataset dengan dimensi tinggi.
- Tidak sensitif terhadap outlier.
- Dapat digunakan untuk klasifikasi dan regresi.

Kerugian  _Support Vector Machine (SVM)_ :
- Sulit untuk memilih kernel dan parameter lainnya. 
- Sensitif terhadap outlier. 
- Membutuhkan banyak waktu komputasi untuk pelatihan.

 _Naïve Bayes Classifier_ merupakan sebuah metoda klasifikasi yang berakar pada teorema Bayes. Metode pengklasifikasian dengan menggunakan metode probabilitas dan statistik yang memprediksi peluang di masa depan berdasarkan pengalaman di masa sebelumnya.
 
 Keuntungan _Naïve Bayes Classifier_:
- Mudah dipahami dan diimplementasikan.
- Cepat untuk dilatih dan diprediksi

Kerugian _Naïve Bayes Classifier_:
- Asumsi independensi fitur mungkin tidak selalu valid.
- Sensitif terhadap fitur dengan nilai nol. 
- Kinerja dapat menurun dengan dataset yang kompleks.

_Extra Trees Classifier_ adalah algoritma machine learning yang digunakan untuk klasifikasi data. Ini mirip dengan Random Forest Classifier yang terkenal, tetapi memiliki beberapa perbedaan utama yaitu _Random Splitting_ dan _No Bagging_. 

keuntungan _Extra Trees Classifier_ :
- Lebih tahan terhadap overfitting dibandingkan dengan Random Forest, terutama pada kumpulan data berdimensi tinggi.
- Mudah diimplementasikan dan digunakan.
- Memiliki kinerja yang baik pada berbagai masalah klasifikasi.

Kerugian _Extra Trees Classifier_ :
- Cenderung kurang akurat dibandingkan Random Forest pada dataset tertentu.
- Membutuhkan banyak waktu komputasi untuk pelatihan.

Parameter yang digunakan adalah:
- `n_estimators` Jumlah pohon keputusan yang akan dibuat dalam ensemble.
- `random_stat`  pengambilan sampel secara acak.
- `max_depth` Kedalaman maksimum pohon keputusan individual.
- `n_jobs` mempercepat pelatihan pada sistem dengan beberapa core CPU.

## Evaluation

Dalam tahap evaluasi, metrik yang digunakan adalah `accuracy`
Accuracy didapatkan dengan menghitung persentase dari jumlah prediksi yang benar dibagi dengan jumlah seluruh prediksi. Rumus:

$$\text{Accuracy} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

*Penjelasan*
- TP (True Positive): Jumlah data positif yang diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (Kesalahan Tipe I).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (Kesalahan Tipe II).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 5 buah model yang latih:

| Model | Accuracy |
| ------ | ------ |
| KNN | 0.90 |
| RandomForest  | 0.89 |
| SVM | 0.89 |
| Naive Bayes | 0.49 |
| Extra Trees Classifier | 0.90 |


Tabel 3. Hasil Accuracy

![Plot Accuracy](https://i.ibb.co/wMPKmm4/akhirkata.png)

Gambar 3. Visualisasi Accuracy Model

Dilihat dari _Tabel 3. Hasil Accuracy_ dan _Gambar 3. Visualisasi Accuracy Model_ tersebut dapat diketahui bahwa model dengan algoritma _KNN_ memiliki Accuracy yang lebih tinggi dengan accuracy `90%` . Untuk itu model tersebut yang akan dipilih untuk digunakan. Diharapkan dengan model yang telah dikembangan dapat memprediksi kualitas apel dengan baik menggunakan _K-Nearest Neighbors (KNN)_. Alasan mengapa metode _KNN_ yang dipilih karena _KNN_ adalah algoritma yang sangat sederhana dibandingkan dengan _Extra Trees Classifier_. Hal ini membuatnya lebih mudah untuk dipahami, diimplementasikan, dan diinterpretasikan. _KNN_ juga tidak memiliki banyak parameter yang perlu dioptimalkan, sehingga lebih mudah untuk digunakan.







## Referensi
[1] Badan Pusat Statistik. (2023). _Produksi Pisang_. [https://www.bps.go.id/id/statistics-table/2/NjIjMg==/produksi-tanaman-buah-buahan.html](https://www.bps.go.id/id/statistics-table/2/NjIjMg==/produksi-tanaman-buah-buahan.html)

[2] Lesmana, G. S., & Murinto. (2022). _Klasifikasi Kualitas Pisang Ambon Menggunakan Metode K-Nearest Neighbor_. Jurnal Sarjana Teknik Informatika, 10(2), 110. [https://doi.org/10.12928/jstie.v10i2.22390](https://doi.org/10.12928/jstie.v10i2.22390)

[3] Rifqi, A., & Butar Butar, B. (2022). _Analisis Kualitas Pisang Berdasarkan Tingkat Kematangan Dengan Algoritma K-Means_. Indonesian Journal on Computer and Information Technology, 7(1), 1–9.[https://repository.bsi.ac.id/repo/files/440819/download/Jurnal_IJCIT_Ahmad-Rifqi.pdf](https://repository.bsi.ac.id/repo/files/440819/download/Jurnal_IJCIT_Ahmad-Rifqi.pdf)


_
