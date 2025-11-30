from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='../templates') # Set folder template

# Konfigurasi Path File (Sesuaikan jika perlu, relatif terhadap lokasi app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'lottery_model.pkl')
FEATURES_PATH = os.path.join(BASE_DIR, 'model_features.pkl')
# Asumsi data ada di ../data/ relative terhadap app.py
DATA_PATH = os.path.join(BASE_DIR, '../data/lottery_test.csv') 

# Load Assets saat aplikasi dimulai
model = None
feature_cols = None
test_df = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH) and os.path.exists(DATA_PATH):
        model = joblib.load(MODEL_PATH)
        feature_cols = joblib.load(FEATURES_PATH)
        test_df = pd.read_csv(DATA_PATH, index_col=0)
        test_df.index = pd.to_datetime(test_df.index)
        print("Model dan data berhasil dimuat.")
    else:
        print("Peringatan: File model atau data tidak ditemukan di path yang ditentukan.")
        print(f"Cek path: {MODEL_PATH}, {FEATURES_PATH}, {DATA_PATH}")

except Exception as e:
    print(f"Error loading files: {e}")

def get_ball_color(number):
    """Menentukan warna bola berdasarkan nomor (Logika Mark Six HK)."""
    reds = [1, 2, 7, 8, 12, 13, 18, 19, 23, 24, 29, 30, 34, 35, 40, 45, 46]
    blues = [3, 4, 9, 10, 14, 15, 20, 25, 26, 31, 36, 37, 41, 42, 47, 48]
    
    if number in reds: return "ball-red"
    elif number in blues: return "ball-blue"
    else: return "ball-green"

def get_prediction(selected_date_str):
    """Melakukan prediksi berdasarkan tanggal."""
    if model is None or test_df is None:
        raise Exception("Model atau data tidak dimuat dengan benar.")
        
    if selected_date_str not in test_df.index:
        raise KeyError("Data tanggal tidak ditemukan.")

    row = test_df.loc[[selected_date_str]] 
    # Pastikan urutan kolom sesuai dengan saat training
    X_input = row[feature_cols]
    
    # Ground Truth (Data Asli)
    y_true_cols = [c for c in test_df.columns if 'num_' in c]
    y_true_row = row[y_true_cols].values.flatten()
    true_balls = np.where(y_true_row == 1)[0] + 1
    
    # Prediksi Probabilitas
    probs = model.predict_proba(X_input)
    # Ambil probabilitas kelas positif (1) untuk setiap bola
    probs_matrix = np.array([p[:, 1] for p in probs]).T
    # Ambil 6 indeks dengan probabilitas tertinggi
    top_6_indices = np.argsort(probs_matrix[0])[-6:][::-1]
    pred_balls = top_6_indices + 1
    
    return sorted(pred_balls), sorted(true_balls)

@app.route('/', methods=['GET', 'POST'])
def index():
    # Persiapan data untuk dropdown tanggal
    years = []
    valid_dates = []

    if test_df is not None:
        dates = test_df.index.sort_values(ascending=False)
        years = sorted(dates.year.unique(), reverse=True)
        valid_dates = [d.strftime('%Y-%m-%d') for d in dates]
    
    prediction_result = None
    error_message = None

    if request.method == 'POST':
        try:
            # Ambil data dari form
            year = int(request.form.get('year'))
            month = int(request.form.get('month'))
            day = int(request.form.get('day'))
            
            # Format tanggal YYYY-MM-DD
            selected_date_str = f"{year}-{month:02d}-{day:02d}"
            
            # Lakukan prediksi
            pred_balls, true_balls = get_prediction(selected_date_str)
            
            # Hitung akurasi
            matches = set(pred_balls).intersection(set(true_balls))
            count_correct = len(matches)
            
            # Siapkan data bola untuk template
            pred_data = [{'num': n, 'color': get_ball_color(n)} for n in pred_balls]
            
            # Asumsi true_balls bisa lebih dari 6 (misal ada special number), ambil 6 pertama sbg main
            true_main = true_balls[:6]
            true_special = true_balls[6] if len(true_balls) > 6 else None
            
            true_data = [{'num': n, 'color': get_ball_color(n)} for n in true_main]
            special_data = {'num': true_special, 'color': 'ball-purple'} if true_special else None

            prediction_result = {
                'date': selected_date_str,
                'pred_balls': pred_data,
                'true_balls': true_data,
                'special_ball': special_data,
                'accuracy': count_correct
            }
            
        except KeyError:
            error_message = "Data untuk tanggal ini tidak ditemukan dalam dataset pengujian."
        except Exception as e:
            error_message = f"Terjadi kesalahan: {str(e)}"

    return render_template('index.html', 
                           years=years, 
                           result=prediction_result, 
                           error=error_message,
                           valid_dates=valid_dates)

if __name__ == '__main__':
    # Jalankan server
    app.run(debug=True, port=5000)