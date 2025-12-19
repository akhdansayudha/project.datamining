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

# --- FUNGSI LOAD DATA ---
@st.cache_resource
def load_and_train_model():
    df = None
    
    # 1. Cari lokasi file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_filenames = [
        "Exam_Score_Prediction.csv", 
        "exam_score_prediction.csv"
    ]
    
    found_path = None
    for filename in possible_filenames:
        full_path = os.path.join(current_dir, filename)
        if os.path.exists(full_path):
            found_path = full_path
            break
            
    if not found_path:
        return None, None, None, None, "❌ File CSV tidak ditemukan. Pastikan nama file 'Exam_Score_Prediction.csv'"

    # 2. LOGIKA "MONTE CARLO": Coba berbagai format, pilih yang datanya paling banyak
    candidates = [] # Menyimpan hasil percobaan pembacaan
    
    # Opsi format yang akan dites (Separator, Encoding)
    options = [
        (';', 'utf-8'),      # Format Excel Indonesia/Eropa (Paling mungkin)
        (',', 'utf-8'),      # Format Standar US
        (';', 'latin-1'),    # Format Excel Lama
        (',', 'latin-1'),    # Format Standar Lama
        ('\t', 'utf-8')      # Tab separated
    ]

    debug_msg = ""

    for sep, enc in options:
        try:
            # Baca tanpa skip baris error (agar kita tahu aslinya)
            temp_df = pd.read_csv(found_path, sep=sep, encoding=enc, on_bad_lines='warn')
            
            # Cek kolom: Bersihkan nama kolom
            temp_df.columns = temp_df.columns.astype(str).str.strip().str.replace('"', '')
            
            # Cek apakah kolom jadi 1 (artinya salah separator)
            if len(temp_df.columns) > 1:
                candidates.append(temp_df)
                debug_msg += f"✅ Sukses baca dengan separator '{sep}' encoding '{enc}': {len(temp_df)} baris.\n"
        except Exception:
            continue

    # 3. PILIH PEMENANG (Dataframe dengan baris terbanyak)
    if not candidates:
        # Jika cara halus gagal, gunakan "Engine Python" (lebih lambat tapi kuat)
        try:
            df = pd.read_csv(found_path, sep=None, engine='python')
        except:
            return None, None, None, None, "❌ Gagal total membaca file. Format file tidak dikenali."
    else:
        # Ambil dataframe yang punya baris paling banyak
        df = max(candidates, key=len)

    # 4. STANDARISASI NAMA KOLOM (PENTING)
    # Mapping nama kolom yang mungkin muncul ke nama standar
    col_map = {c: c.lower().replace(' ', '_').replace('-', '_') for c in df.columns}
    target_col = None
    
    # Cari kolom target 'exam_score' atau 'score'
    for original, clean in col_map.items():
        if clean == 'exam_score' or clean == 'score': 
            target_col = original
            break
            
    if target_col:
        df.rename(columns={target_col: 'exam_score'}, inplace=True)
    else:
        return None, None, None, None, f"❌ Kolom 'exam_score' tidak ditemukan. Kolom yang terbaca: {list(df.columns)}"

    # 5. PEMBERSIHAN DATA (CLEANING)
    # Paksa exam_score jadi angka (data string/teks akan jadi NaN)
    df['exam_score'] = pd.to_numeric(df['exam_score'], errors='coerce')
    
    # Hapus baris yang nilainya kosong/rusak
    df.dropna(subset=['exam_score'], inplace=True)
    
    # --- PROSES TRAINING MODEL ---
    try:
        df_clean = df.copy()
        
        # Target Variable (Lulus jika >= 75)
        df_clean["Status"] = np.where(df_clean["exam_score"] >= 75, 1, 0)
        
        encoders = {}
        cat_columns = df_clean.select_dtypes(include=['object']).columns
        
        for col in cat_columns:
            le = LabelEncoder()
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            encoders[col] = le

        # Drop kolom yang tidak dipakai untuk prediksi
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

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Navigasi")
menu = st.sidebar.radio("Pilih Menu:", ["🏠 Beranda", "📊 Analisis Data", "🤖 Prediksi Kelulusan"])
st.sidebar.markdown("---")

# --- HALAMAN BERANDA ---
if menu == "🏠 Beranda":
    st.title("🎓 Sistem Analisis & Prediksi Nilai Ujian")
    
    if status_msg is not None and isinstance(status_msg, str):
        st.error("⚠️ TERJADI MASALAH MEMUAT DATA")
        st.code(status_msg)
        st.warning("Solusi: Pastikan file CSV tidak dibuka di Excel saat menjalankan aplikasi.")
    
    elif data_raw is not None:
        st.markdown(f"""
        Selamat datang di **Exam Score Prediction Dashboard**. 
        Data diproses secara otomatis meliputi *Encoding Kategori*, *Scaling*, dan *Pemodelan Klasifikasi*.
        """)
        
        st.success("✅ Data berhasil terhubung!")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Siswa", f"{len(data_raw):,}")
        col2.metric("Rata-rata Nilai", f"{data_raw['exam_score'].mean():.2f}")
        pass_count = len(data_raw[data_raw['exam_score'] >= 75])
        col3.metric("Tingkat Kelulusan", f"{(pass_count/len(data_raw)*100):.1f}%")

# --- HALAMAN ANALISIS DATA ---
elif menu == "📊 Analisis Data":
    st.title("📊 Visualisasi & Analisis Data")
    
    if data_raw is not None:
        tab1, tab2, tab3 = st.tabs(["Data Overview", "Distribusi & Relasi", "Faktor Penentu"])
        
        with tab1:
            st.subheader("Dataset Mentah (Sampel 100 Data)")
            st.dataframe(data_raw.head(100))
            st.subheader("Statistik Deskriptif")
            st.write(data_raw.describe())
        
        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Distribusi Nilai Ujian")
                fig_hist = px.histogram(data_raw, x="exam_score", nbins=50, title="Sebaran Skor Ujian", color_discrete_sequence=['#636EFA'])
                st.plotly_chart(fig_hist, use_container_width=True)
            with col_b:
                st.subheader("Proporsi Kelulusan (Score >= 75)")
                data_raw['Status_Label'] = np.where(data_raw['exam_score'] >= 75, 'Lulus', 'Tidak Lulus')
                fig_pie = px.pie(data_raw, names='Status_Label', title="Persentase Kelulusan", hole=0.4, color_discrete_sequence=['#EF553B', '#00CC96'])
                st.plotly_chart(fig_pie, use_container_width=True)

            st.subheader("Hubungan Jam Belajar vs Nilai Ujian")
            fig_scatter = px.scatter(data_raw, x="study_hours", y="exam_score", color="Status_Label", 
                                     title="Scatter Plot: Study Hours vs Exam Score", opacity=0.6)
            st.plotly_chart(fig_scatter, use_container_width=True)

        with tab3:
            st.subheader("Rata-rata Nilai Berdasarkan Kategori")
            cat_option = st.selectbox("Pilih Kategori:", ["study_method", "parental_education", "gender", "course", "sleep_quality"], index=0)
            if cat_option in data_raw.columns:
                avg_score = data_raw.groupby(cat_option)['exam_score'].mean().reset_index().sort_values(by='exam_score', ascending=False)
                fig_bar = px.bar(avg_score, x=cat_option, y='exam_score', color='exam_score', title=f"Rata-rata Skor per {cat_option}")
                st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.error("Data belum dimuat. Cek halaman Beranda untuk detail error.")

# --- HALAMAN PREDIKSI ---
elif menu == "🤖 Prediksi Kelulusan":
    st.title("🤖 Cek Prediksi Kelulusan Siswa")
    
    if model is not None:
        with st.form("prediction_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.subheader("Data Demografis")
                age = st.number_input("Umur", min_value=15, max_value=30, value=20)
                # Ambil opsi unik dari data asli untuk dropdown
                gender = st.selectbox("Gender", options=np.unique(data_raw['gender'].astype(str)))
                course = st.selectbox("Jurusan (Course)", options=np.unique(data_raw['course'].astype(str)))
            with col2:
                st.subheader("Data Akademik")
                study_hours = st.slider("Jam Belajar (per hari)", 0.0, 15.0, 4.0)
                attendance = st.slider("Kehadiran Kelas (%)", 0.0, 100.0, 75.0)
                method = st.selectbox("Metode Belajar", options=np.unique(data_raw['study_method'].astype(str)))
                facility = st.selectbox("Rating Fasilitas", options=np.unique(data_raw['facility_rating'].astype(str)))
            with col3:
                st.subheader("Gaya Hidup & Lainnya")
                sleep_hours = st.slider("Jam Tidur", 0.0, 12.0, 7.0)
                sleep_qual = st.selectbox("Kualitas Tidur", options=np.unique(data_raw['sleep_quality'].astype(str)))
                internet = st.selectbox("Akses Internet", options=np.unique(data_raw['internet_access'].astype(str)))
                difficulty = st.selectbox("Tingkat Kesulitan Ujian", options=np.unique(data_raw['exam_difficulty'].astype(str)))
            
            submit_btn = st.form_submit_button("🔍 Prediksi Sekarang")
        
        if submit_btn:
            input_data = pd.DataFrame({
                'age': [age], 'gender': [gender], 'course': [course],
                'study_hours': [study_hours], 'class_attendance': [attendance],
                'internet_access': [internet], 'sleep_hours': [sleep_hours],
                'sleep_quality': [sleep_qual], 'study_method': [method],
                'facility_rating': [facility], 'exam_difficulty': [difficulty]
            })
            
            # Transformasi input user menggunakan encoder yang sudah dilatih
            for col, le in encoders.items():
                # Handle unknown labels (jika user input sesuatu yang tidak ada di training data)
                input_data[col] = input_data[col].astype(str)
                # (Disini kita asumsi input dari dropdown pasti ada di encoder)
                input_data[col] = le.transform(input_data[col])
            
            X_cols = scaler.feature_names_in_
            input_data = input_data[X_cols]
            input_scaled = scaler.transform(input_data)
            
            prediction = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0]
            
            st.divider()
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                if prediction == 1:
                    st.success("### ✅ HASIL: LULUS")
                    st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=150)
                else:
                    st.error("### ❌ HASIL: TIDAK LULUS")
                    st.image("https://cdn-icons-png.flaticon.com/512/1828/1828843.png", width=150)
            with col_res2:
                st.write("#### Detail Probabilitas:")
                st.progress(int(prob[1]*100))
                st.caption(f"Probabilitas Lulus: {prob[1]*100:.2f}%")
                st.info(f"Probabilitas Lulus: **{prob[1]*100:.1f}%**")
    else:
         st.error("Model belum siap. Data tidak terbaca.")

# Footer
st.markdown("---")
st.caption("Dikembangkan oleh kelompok 🌲🌲🌲")