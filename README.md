# Sistem Klasifikasi Kelayakan Calon Pendonor Darah Menggunakan Algoritma Naïve Bayes

Aplikasi berbasis web untuk membantu proses klasifikasi kelayakan calon pendonor darah berdasarkan data identitas, pemeriksaan fisik, dan riwayat kesehatan. Sistem ini mengimplementasikan algoritma **Naïve Bayes** sebagai metode machine learning untuk mengklasifikasikan calon pendonor ke dalam dua kategori, yaitu **Layak** dan **Tidak Layak**.

Proyek ini dikembangkan sebagai implementasi machine learning dalam sistem informasi berbasis web dengan studi kasus PMI Kota Bandung.

## 🎯 Latar Belakang

Proses skrining calon pendonor darah merupakan tahapan penting untuk memastikan bahwa pendonor memenuhi persyaratan kesehatan sebelum melakukan donor darah.

Aplikasi ini dikembangkan untuk membantu proses klasifikasi berdasarkan sejumlah parameter pemeriksaan, sehingga petugas dapat memperoleh hasil prediksi kelayakan secara lebih terstruktur.

## ✨ Fitur Utama

- **Autentikasi Pengguna** — Login dan registrasi untuk petugas.
- **Input Data Pendonor** — Pencatatan identitas dan informasi calon pendonor.
- **Pemeriksaan Fisik** — Input usia, berat badan, hemoglobin, tekanan darah, denyut nadi, dan suhu tubuh.
- **Riwayat Kesehatan** — Pencatatan riwayat operasi, kehamilan, demam, transfusi darah, perilaku seksual berisiko, dan penyakit kronis.
- **Klasifikasi Naïve Bayes** — Prediksi kelayakan calon pendonor menggunakan model machine learning.
- **Validasi Standar Skrining** — Pemeriksaan sejumlah parameter berdasarkan aturan kelayakan yang diterapkan dalam aplikasi.
- **Riwayat Klasifikasi** — Penyimpanan hasil prediksi calon pendonor ke dalam database.
- **Evaluasi Model** — Penyajian informasi evaluasi performa model.

## 🧠 Algoritma Machine Learning

Sistem ini menggunakan algoritma **Naïve Bayes**, yaitu metode klasifikasi probabilistik berdasarkan Teorema Bayes dengan asumsi independensi antarfitur.

Model machine learning digunakan untuk mengklasifikasikan data calon pendonor berdasarkan fitur-fitur pemeriksaan yang telah melalui tahap preprocessing.

Tahapan umum pengolahan data meliputi:

1. Data Collection
2. Data Preprocessing
3. Data Transformation
4. Model Training
5. Model Evaluation
6. Model Deployment

Model yang telah dilatih disimpan dalam format `.pkl` menggunakan Joblib untuk kemudian digunakan oleh aplikasi Flask.

## 🛠️ Teknologi yang Digunakan

| Teknologi | Kegunaan |
|---|---|
| Python | Bahasa pemrograman utama |
| Flask | Framework backend aplikasi web |
| Scikit-learn | Implementasi machine learning |
| Pandas | Pengolahan dan preprocessing data |
| NumPy | Operasi komputasi numerik |
| Joblib | Penyimpanan dan pemuatan model |
| MySQL | Database penyimpanan data |
| Flask-SQLAlchemy | ORM untuk interaksi database |
| Flask-Login | Autentikasi dan manajemen sesi |
| HTML | Struktur halaman web |
| Tailwind CSS | Styling dan antarmuka pengguna |
| JavaScript | Interaksi dan komunikasi frontend dengan backend |
| Chart.js | Visualisasi data dan evaluasi model |
| DataTables | Pengelolaan dan penyajian tabel data |

## 📋 Persyaratan Sistem

Sebelum menjalankan aplikasi, pastikan perangkat telah memiliki:

- Python 3.10 atau versi yang kompatibel dengan dependensi proyek.
- MySQL Server.
- Git.
- Web browser.
- Pip untuk instalasi package Python.

## ⚙️ Instalasi dan Konfigurasi

### 1. Clone Repository

```bash
git clone https://github.com/USERNAME/REPOSITORY.git
```

Masuk ke direktori proyek:

```bash
cd REPOSITORY
```

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment.

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

Jika repository menyediakan `requirements.txt`, jalankan:

```bash
pip install -r requirements.txt
```

### 4. Konfigurasi Database

Buat database MySQL dengan nama:

```sql
CREATE DATABASE skripsi_pmi;
```

Sesuaikan konfigurasi koneksi database pada aplikasi dengan kredensial MySQL di lingkungan lokal Anda.

### 5. Siapkan Model Machine Learning

Pastikan file model berikut tersedia pada lokasi yang sesuai dengan konfigurasi aplikasi:

```text
model_naive_bayes_pmi.pkl
```

File tersebut berisi model Naïve Bayes dan encoder yang digunakan untuk preprocessing data sebelum klasifikasi.

### 6. Jalankan Aplikasi

```bash
python app.py
```

Setelah server berjalan, buka browser dan akses:

```text
http://127.0.0.1:5000
```

## 📂 Struktur Fitur Aplikasi

Secara umum, aplikasi mencakup beberapa bagian utama:

- **Dashboard** — Menampilkan informasi ringkasan aplikasi.
- **Form Skrining** — Digunakan untuk memasukkan data calon pendonor.
- **Klasifikasi** — Menampilkan hasil prediksi kelayakan.
- **Riwayat Klasifikasi** — Menyimpan dan menampilkan hasil klasifikasi sebelumnya.
- **Evaluasi Model** — Menampilkan informasi performa model machine learning.

## 📊 Kategori Hasil Klasifikasi

Sistem menghasilkan dua kategori klasifikasi:

| Kategori | Keterangan |
|---|---|
| Layak | Calon pendonor diklasifikasikan memenuhi kriteria berdasarkan sistem. |
| Tidak Layak | Calon pendonor diklasifikasikan tidak memenuhi kriteria berdasarkan sistem. |

**Catatan:** Hasil klasifikasi merupakan rekomendasi pendukung proses skrining dan bukan pengganti pemeriksaan maupun keputusan medis petugas yang berwenang.

## 🔒 Keamanan dan Privasi

Aplikasi mengelola informasi identitas dan data kesehatan calon pendonor. Oleh karena itu, penggunaan aplikasi perlu memperhatikan:

- Perlindungan informasi pribadi calon pendonor.
- Pengamanan kredensial database.
- Pengelolaan hak akses pengguna.
- Keamanan penyimpanan dan akses data klasifikasi.

Hindari mengunggah data pendonor asli, kredensial database, maupun informasi sensitif ke repository publik.

## 👨‍💻 Pengembang

**Hikam Sirrul Arifin**

Informatics Engineering Graduate | Full-Stack Web Developer

Bidang minat:
- Software Development
- Full-Stack Web Development
- Machine Learning
- Cybersecurity

## 📄 Lisensi

Proyek ini dikembangkan untuk keperluan akademik dan portofolio. Penggunaan, pengembangan, atau distribusi lebih lanjut dapat disesuaikan dengan ketentuan lisensi yang ditetapkan oleh pengembang.

---

**Sistem Klasifikasi Kelayakan Calon Pendonor Darah Menggunakan Algoritma Naïve Bayes Berbasis Web — Studi Kasus PMI Kota Bandung.**
