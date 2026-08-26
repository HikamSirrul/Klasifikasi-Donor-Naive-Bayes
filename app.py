import flask
from flask import Flask, request, jsonify, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime
import joblib
import pandas as pd
import numpy as np
import os
from io import BytesIO
from flask import send_file

app = Flask(__name__)
app.config['SECRET_KEY'] = 'rahasia_skripsi_pmi_2025'

# --- KONEKSI DATABASE MYSQL ---
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/skripsi_pmi'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# --- MODEL DATABASE ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nama_lengkap = db.Column(db.String(150))
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(200), nullable=False)
    role = db.Column(db.String(50), default='Petugas')

class Riwayat(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    tanggal = db.Column(db.DateTime, default=datetime.now)
    petugas = db.Column(db.String(100))
    # Identitas
    nama_pendonor = db.Column(db.String(100))
    nik = db.Column(db.String(20))
    alamat = db.Column(db.Text)
    gol_darah = db.Column(db.String(20))
    # Fisik
    usia = db.Column(db.Integer)
    jenis_kelamin = db.Column(db.String(20))
    berat_badan = db.Column(db.Float)
    hb = db.Column(db.Float)
    tensi_sistolik = db.Column(db.Integer)
    tensi_diastolik = db.Column(db.Integer)
    nadi = db.Column(db.Integer)
    suhu = db.Column(db.Float)
    # Risiko
    riwayat_operasi = db.Column(db.String(10))
    riwayat_hamil = db.Column(db.String(20))
    sakit_kepala = db.Column(db.String(10))
    transfusi = db.Column(db.String(10))
    seks_berisiko = db.Column(db.String(10))
    penyakit_kronis = db.Column(db.String(10))
    # Kesiapan
    tidur_cukup = db.Column(db.String(10))
    makan_sebelum = db.Column(db.String(10))
    # Hasil
    status_prediksi = db.Column(db.String(50))
    alasan_penolakan = db.Column(db.Text)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# --- LOAD MODEL NB ---
try:
    # Load file 
    paket = joblib.load('model_naive_bayes_pmi.pkl')
    
    # Ekstrak komponen
    model = paket['model']
    encoder = paket['encoder']
    
        
    print(f">>> SUKSES: Model Naive Bayes Dimuat")
except Exception as e:
    print(f">>> ERROR: Gagal memuat file model Naive Bayes! {e}")

# --- PREPROCESSING ---
ATURAN_BINNING = {
    'Usia': {'bins': [-1, 16, 60, 150], 'labels': ['Dibawah_Umur', 'Usia_Produktif', 'Diatas_Umur']},
    'BB': {'bins': [-1, 44.9, 300], 'labels': ['Terlalu_Ringan', 'Ideal']},
    'Sistolik': {'bins': [-1, 89, 160, 400], 'labels': ['Rendah', 'Normal', 'Tinggi']},
    'Diastolik': {'bins': [-1, 59, 100, 300], 'labels': ['Rendah', 'Normal', 'Tinggi']},
    'Nadi': {'bins': [-1, 49, 100, 300], 'labels': ['Rendah', 'Normal', 'Tinggi']},
    'Suhu': {'bins': [-1, 36.4, 37.5, 60], 'labels': ['Rendah', 'Normal', 'Tinggi']},
    'HB': {'bins': [-1, 12.4, 17.0, 30], 'labels': ['Rendah', 'Normal', 'Tinggi']}
}

KOLOM_FITUR = [
    'Usia', 'Jenis_Kelamin', 'BB', 'Sistolik', 'Diastolik', 
    'Nadi', 'Suhu', 'HB', 
    'Operasi', 'Kehamilan', 
    'Demam', 'Transfusi', 
    'Seksual_Berisko',
    'Penyakit_Kronis',  
    'Tidur_Cukup', 'Sarapan'
]


STANDAR_PMI = {
    'Usia': {'min': 17, 'max': 60},        
    'Nadi': {'min': 50, 'max': 100},       
    'Suhu': {'min': 36.5, 'max': 37.5},    
}


def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    for col, rule in ATURAN_BINNING.items():
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce').fillna(0)
        df_input[col] = pd.cut(df_input[col], bins=rule['bins'], labels=rule['labels'], right=True)
        if df_input[col].isnull().any(): df_input[col] = rule['labels'][0]

    # Filter hanya kolom yang dibutuhkan model
    # Jika model butuh kolom yg tidak ada di form, kita isi default "Tidak"
    for col in KOLOM_FITUR:
        if col not in df_input.columns:
            df_input[col] = "Tidak"

    try:
        df_ordered = df_input[KOLOM_FITUR]
        data_encoded = encoder.transform(df_ordered.astype(str))
        data_encoded[data_encoded < 0] = 0
        return data_encoded
    except KeyError as e:
        raise ValueError(f"Input Kurang: {e}")


def cek_standar_pmi(data):

    try:
        usia = float(data.get('Usia', 0))
        batas_usia = STANDAR_PMI['Usia']
        if usia < batas_usia['min'] or usia > batas_usia['max']:
            return False
    except (ValueError, TypeError):
        return False

    try:
        nadi = float(data.get('Nadi', 0))
        batas_nadi = STANDAR_PMI['Nadi']
        if nadi < batas_nadi['min'] or nadi > batas_nadi['max']:
            return False
    except (ValueError, TypeError):
        return False

    try:
        suhu = float(data.get('Suhu', 0))
        batas_suhu = STANDAR_PMI['Suhu']
        if suhu < batas_suhu['min'] or suhu > batas_suhu['max']:
            return False
    except (ValueError, TypeError):
        return False

    return True


def cari_alasan_penolakan(data):
    alasan = []
    try:
        if float(data.get('Usia',0)) < 17: alasan.append("Usia < 17")
        if float(data.get('Usia',0)) > 60: alasan.append("Usia > 60")
        if float(data.get('BB',0)) < 49.9: alasan.append("BB < 45 kg")
        if float(data.get('HB',0)) < 12.5: alasan.append("Hb Rendah")
        if float(data.get('Sistolik',0)) > 160: alasan.append("Tensi Tinggi")
        if float(data.get('Sistolik',0)) < 100: alasan.append("Tensi Rendah")
        if float(data.get('Diastolik',0)) > 100: alasan.append("Tensi Tinggi")
        if float(data.get('Diastolik',0)) < 60: alasan.append("Tensi Rendah")
        if float(data.get('Nadi',0)) < 50: alasan.append("Nadi < 50 x/menit")
        if float(data.get('Nadi',0)) > 100: alasan.append("Nadi > 100 x/menit")
        if float(data.get('Suhu',0)) < 36.5: alasan.append("Suhu < 36.5")
        if float(data.get('Suhu',0)) > 37.5: alasan.append("Suhu > 37.5")
    except: pass
    
    # List risiko 
    risk = []
    for r in risk:
        if data.get(r) == 'Ya': alasan.append(f"{r.replace('_', ' ')}")
    if data.get('Operasi') == 'Ya': alasan.append("Riwayat Operasi < 6 Bulan")
    if data.get('Demam') == 'Ya': alasan.append("Sedang Demam")
    if data.get('Transfusi') == 'Ya': alasan.append("Menerima Transfusi Darah < 6 Bulan Terakhir")
    if data.get('Seksual_Berisko') == 'Ya': alasan.append("Perilaku Seksual Berisiko Tinggi")
    if data.get('Penyakit_Kronis') == 'Ya': alasan.append("Riwayat Penyakit Kronis")
    if data.get('Kehamilan') == 'Ya': alasan.append("Sedang Hamil/Menyusui")
    if data.get('Tidur_Cukup') == 'Tidak': alasan.append("Kurang Tidur")
    if data.get('Sarapan') == 'Tidak': alasan.append("Tidak Makan Sebelum Donor")

    return alasan if alasan else ["Analisis Model"]

# --- ROUTES ---
@app.route('/')
def root(): return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated: return redirect(url_for('dashboard'))
    if request.method == 'POST':
        user = User.query.filter_by(username=request.form['username']).first()
        if user and check_password_hash(user.password, request.form['password']):
            login_user(user)
            return redirect(url_for('dashboard'))
        flash('Username atau Password salah!', 'error')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        uname = request.form['username']
        pw = request.form['password']
        nama = request.form['nama']
        if User.query.filter_by(username=uname).first():
            flash('Username sudah dipakai!', 'error')
        else:
            new_user = User(username=uname, nama_lengkap=nama,
                            password=generate_password_hash(pw, method='pbkdf2:sha256'))
            db.session.add(new_user)
            db.session.commit()
            flash('Registrasi Berhasil! Silakan Login.', 'success')
            return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/dashboard')
@login_required
def dashboard():
    # 1. Ambil Data Riwayat (Untuk Tabel)
    data_riwayat = Riwayat.query.order_by(Riwayat.id.desc()).all()

    # 2. Hitung Statistik Total dari Database (BARU)
    total_layak = Riwayat.query.filter_by(status_prediksi='Layak').count()
    total_tidak = Riwayat.query.filter_by(status_prediksi='Tidak Layak').count()
    total_semua = total_layak + total_tidak

    # Bungkus dalam dictionary agar rapi saat dikirim ke HTML
    statistik = {
        'layak': total_layak,
        'tidak': total_tidak,
        'total': total_semua
    }

    return render_template('dashboard.html', user=current_user, riwayat=data_riwayat, stats=statistik)

# --- ROUTE DELETE RIWAYAT ---
@app.route('/delete_riwayat/<int:id>', methods=['POST'])
@login_required
def delete_riwayat(id):
    try:
        riwayat = Riwayat.query.get_or_404(id)
        db.session.delete(riwayat)
        db.session.commit()
        # Ganti kategori jadi 'warning' agar bisa kita bedakan warnanya nanti
        flash('Data riwayat berhasil dihapus.', 'warning') 
    except:
        flash('Gagal menghapus data.', 'error')
    return redirect(url_for('dashboard'))

@app.route('/delete_all_riwayat', methods=['POST'])
@login_required
def delete_all_riwayat():
    try:
        num_rows = db.session.query(Riwayat).delete()
        db.session.commit()
        # Ganti kategori jadi 'warning' (Merah)
        flash(f'Berhasil menghapus {num_rows} data riwayat.', 'warning')
    except:
        flash('Gagal menghapus semua data.', 'error')
    return redirect(url_for('dashboard'))

# --- ROUTE EXPORT EXCEL ---
@app.route('/export_excel')
@login_required
def export_excel():
    try:
        riwayat_data = Riwayat.query.order_by(Riwayat.id.desc()).all()
        if not riwayat_data:
            flash("Belum ada data untuk diexport.", "error")
            return redirect(url_for('dashboard'))

        # 2. Konversi data database ke format List of Dictionaries
        data_list = []
        for row in riwayat_data:
            data_list.append({
                'ID': row.id,
                'Tanggal': row.tanggal.strftime('%Y-%m-%d %H:%M:%S'),
                'Petugas': row.petugas,
                'Nama_Pendonor': row.nama_pendonor,
                'NIK': row.nik,
                'Alamat': row.alamat,
                'Golongan_Darah': row.gol_darah,
                'Usia': row.usia,
                'Jenis_Kelamin': row.jenis_kelamin,
                'BB': row.berat_badan,
                'HB': row.hb,
                'Sistolik': row.tensi_sistolik,
                'Diastolik': row.tensi_diastolik,
                'Nadi': row.nadi,
                'Suhu': row.suhu,
                'Operasi': row.riwayat_operasi,
                'RiwayatKehamilan': row.riwayat_hamil,
                'Demam': row.sakit_kepala,
                'Transfusi': row.transfusi,
                'Seksual_Berisko': row.seks_berisiko,
                'Penyakit_Kronis': row.penyakit_kronis,
                'Tidur_Cukup': row.tidur_cukup,
                'Sarapan': row.makan_sebelum,
                'Status_Prediksi': row.status_prediksi,
                'Alasan_Penolakan': row.alasan_penolakan
            }),
        # 3. Buat DataFrame Pandas
        df_export = pd.DataFrame(data_list)

        # 4. Tulis ke Excel (In-Memory Buffer)
        output = BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df_export.to_excel(writer, index=False, sheet_name='Riwayat Skrining')

        # Reset pointer buffer ke awal
        output.seek(0)

        # 5. Kirim file ke browser user
        filename = f"Laporan_Donor_PMI_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"

        return send_file(
            output,
            mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            as_attachment=True,
            download_name=filename
        )

    except Exception as e:
        print(f"Error Export: {e}")
        flash(f"Gagal export data: {str(e)}", "error")
        return redirect(url_for('dashboard'))

@app.route('/predict_page')
@login_required
def predict_page(): return render_template('predict.html')

# --- ROUTE EVALUASI ---
@app.route('/evaluasi')
@login_required
def evaluasi():
    metrics = {
        'training_accuracy': 94.76,
        'testing_accuracy': 94.33,
        'presisi': 94,
        'recall': 94,
        'f1_score': 94
    }
    return render_template('evaluasi.html', metrics=metrics)

@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    if not model: return jsonify({'error': 'Model Error'}), 500
    try:
        data = request.json
        processed = preprocess_input(data)
        pred = model.predict(processed)[0]
        hasil = 'Layak' if pred == 0 else 'Tidak Layak'

        if not cek_standar_pmi(data):
            hasil = 'Tidak Layak'

        # Satu-satunya sumber teks alasan penolakan
        alasan = cari_alasan_penolakan(data) if hasil == 'Tidak Layak' else []
        alasan_str = ", ".join(alasan) if alasan else "-"

        # Simpan ke DB
        riwayat_baru = Riwayat(
            petugas=current_user.nama_lengkap,
            nama_pendonor=data.get('Nama_Pendonor'), nik=data.get('NIK'),
            alamat=data.get('Alamat'), gol_darah=data.get('Golongan_Darah'),
            usia=data.get('Usia'), jenis_kelamin=data.get('Jenis_Kelamin'),
            berat_badan=data.get('BB'), hb=data.get('HB'),
            tensi_sistolik=data.get('Sistolik'), tensi_diastolik=data.get('Diastolik'),
            suhu=data.get('Suhu'), nadi=data.get('Nadi'),
            # Kuesioner
            riwayat_operasi=data.get('Operasi'), riwayat_hamil=data.get('Kehamilan'),
            sakit_kepala=data.get('Demam'), transfusi=data.get('Transfusi'),
            seks_berisiko=data.get('Seksual_Berisko'), penyakit_kronis=data.get('Penyakit_Kronis'),
            # Kesiapan
            tidur_cukup=data.get('Tidur_Cukup'), makan_sebelum=data.get('Sarapan'),
            status_prediksi=hasil, alasan_penolakan=alasan_str
        )
        db.session.add(riwayat_baru)
        db.session.commit()
        return jsonify({'prediksi': hasil, 'alasan': alasan})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True, port=5000)