import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

# Konfigurasi Halaman Website
st.set_page_config(
    page_title="Sistem Prediksi Nilai Ujian",
    page_icon="🎓",
    layout="wide"
)

# --- FUNGSI LOAD DATA (FIXED PATH: Root -> data/raw/) ---
@st.cache_resource
def load_and_train_model():
    df = None
    
    # 1. TENTUKAN LOKASI FILE
    # Mendapatkan lokasi file app.py ini berada (Root folder)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Target: masuk ke folder 'data' -> 'raw' -> nama file
    target_path = os.path.join(base_dir, "data", "raw", "Exam_Score_Prediction.csv")
    
    # Cek apakah file benar-benar ada di sana
    if not os.path.exists(target_path):
        return None, None, None, None, f"❌ File tidak ditemukan.\n\nSistem mencari di:\n{target_path}\n\nPastikan folder 'data' dan 'raw' sudah benar."

    # 2. BACA DATA (Metode Smart Reader)
    candidates = [] 
    
    # Opsi: Coba Koma (,), Titik Koma (;), dan Auto-Engine
    options = [
        (',', 'utf-8', 'c'),       # Standard
        (';', 'utf-8', 'c'),       # Excel Indo
        (None, 'utf-8', 'python'), # Auto-detect
    ]

    for sep, enc, eng in options:
        try:
            if sep is None:
                temp_df = pd.read_csv(target_path, sep=None, engine='python', encoding=enc)
            else:
                temp_df = pd.read_csv(target_path, sep=sep, encoding=enc, engine=eng, on_bad_lines='warn')
            
            # Bersihkan nama kolom
            temp_df.columns = temp_df.columns.astype(str).str.strip().str.replace('"', '')
            
            # Validasi: Harus lebih dari 1 kolom
            if len(temp_df.columns) > 1:
                candidates.append(temp_df)
        except Exception:
            continue

    # 3. PILIH DATA TERBAIK
    if not candidates:
        return None, None, None, None, "❌ Gagal membaca file. Format CSV rusak/tidak dikenali."
    
    # Ambil yang barisnya paling banyak (mendekati 20.000)
    df = max(candidates, key=len)

    # 4. RENAME KOLOM TARGET
    col_map = {c: c.lower().replace(' ', '_').replace('-', '_').replace('.', '_') for c in df.columns}
    target_col = None
    
    for original, clean in col_map.items():
        if 'exam_score' in clean or 'score' in clean: 
            target_col = original
            break
            
    if target_col:
        df.rename(columns={target_col: 'exam_score'}, inplace=True)
    else:
        return None, None, None, None, f"❌ Kolom 'exam_score' tidak ditemukan.\nKolom: {list(df.columns)}"

    # 5. BERSIHKAN DATA (Hapus Baris Kosong)
    df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
    df.dropna(subset=['exam_score'], inplace=True)
    
    # --- TRAINING MODEL ---
    try:
        df_clean = df.copy()
        df_clean["Status"] = np.where(df_clean["exam_score"] >= 75, 1, 0)
        
        encoders = {}
        cat_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in cat_columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le

        drop_cols = ["student_id", "exam_score", "Status"]
        existing_drop_cols = [c for c in drop_cols if c in df_clean.columns]
        
        X = df_clean.drop(columns=existing_drop_cols)
        y = df_clean["Status"]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        return df, model, encoders, scaler, acc

    except Exception as e:
        return None, None, None, None, f"❌ Error Training: {str(e)}"

# Load Resources
data_raw, model, encoders, scaler, status_msg = load_and_train_model()

# --- INTERFACE STREAMLIT ---
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Beranda", "📊 Analisis Data", "🤖 Prediksi Kelulusan"])
st.sidebar.markdown("---")

if menu == "🏠 Beranda":
    st.title("🎓 Sistem Analisis & Prediksi Nilai Ujian")
    
    if status_msg and isinstance(status_msg, str):
        st.error("⚠️ GAGAL MEMUAT DATA")
        st.code(status_msg)
    elif data_raw is not None:
        st.success("✅ Data Berhasil Dimuat!")
        st.caption(f"Sumber Data: data/raw/Exam_Score_Prediction.csv")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Siswa", f"{len(data_raw):,}")
        c2.metric("Rata-rata Nilai", f"{data_raw['exam_score'].mean():.2f}")
        pass_rate = (len(data_raw[data_raw['exam_score'] >= 75]) / len(data_raw)) * 100
        c3.metric("Kelulusan", f"{pass_rate:.1f}%")

elif menu == "📊 Analisis Data":
    st.title("📊 Visualisasi Data")
    if data_raw is not None:
        tab1, tab2 = st.tabs(["Tabel Data", "Grafik"])
        with tab1:
            st.dataframe(data_raw.head(100))
        with tab2:
            st.subheader("Distribusi Skor")
            fig = px.histogram(data_raw, x="exam_score", nbins=50)
            st.plotly_chart(fig, use_container_width=True)
            
            # Cek kolom study_hours
            s_col = None
            for c in data_raw.columns:
                if 'study' in c.lower() and 'hour' in c.lower(): s_col = c
            
            if s_col:
                st.subheader(f"Hubungan {s_col} vs Score")
                fig2 = px.scatter(data_raw, x=s_col, y="exam_score")
                st.plotly_chart(fig2, use_container_width=True)
    else:
        st.error("Data belum siap.")

elif menu == "🤖 Prediksi Kelulusan":
    st.title("🤖 Prediksi Kelulusan")
    if model:
        with st.form("pred_form"):
            cols = st.columns(3)
            inputs = {}
            # Auto-generate inputs based on training features
            features = scaler.feature_names_in_
            
            for i, col in enumerate(features):
                with cols[i % 3]:
                    if col in encoders:
                        # Dropdown untuk kategori
                        opts = sorted(list(encoders[col].classes_))
                        inputs[col] = st.selectbox(col, opts)
                    else:
                        # Input angka
                        inputs[col] = st.number_input(col, value=0)
            
            submit = st.form_submit_button("🔍 Prediksi")
        
        if submit:
            input_df = pd.DataFrame([inputs])
            # Encode & Scale
            for col, le in encoders.items():
                if col in input_df:
                    input_df[col] = le.transform(input_df[col].astype(str))
            
            scaled = scaler.transform(input_df)
            pred = model.predict(scaled)[0]
            prob = model.predict_proba(scaled)[0][1]
            
            st.divider()
            if pred == 1:
                st.success(f"### ✅ HASIL: LULUS (Peluang: {prob*100:.1f}%)")
            else:
                st.error(f"### ❌ HASIL: TIDAK LULUS (Peluang: {prob*100:.1f}%)")
    else:
        st.error("Model gagal dimuat.")

st.markdown("---")
st.caption("Dikembangkan oleh kelompok 🌲🌲🌲")