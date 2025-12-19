import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
sys.path.append(SRC_DIR)

from data_preparation import load_data, preprocess_data, save_processed_data
from clustering import elbow_method, kmeans_clustering
from classification import create_target, train_classification
import os
import streamlit as st
st.title("Streamlit Test")
st.success("App berhasil dijalankan 🎉")

# =========================
# PATH CONFIGURATION
# =========================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "Exam_Score_Prediction.csv.xlsx"
)

PROCESSED_DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "exam_score_processed.xlsx"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "output",
    "results",
    "hasil_analisis_exam_score.xlsx"
)


# =========================
# MAIN PIPELINE
# =========================

def main():
    print("=== DATA MINING PIPELINE STARTED ===\n")

    # 1. Load Data
    print("1. Loading raw data...")
    df = load_data(RAW_DATA_PATH)
    print("   Data loaded successfully\n")

    # 2. Data Preparation
    print("2. Data preprocessing...")
    df_clean, scaled_data = preprocess_data(df)

    # Pastikan folder data/processed ada
    os.makedirs(os.path.dirname(PROCESSED_DATA_PATH), exist_ok=True)
    save_processed_data(df_clean, PROCESSED_DATA_PATH)
    print("   Processed data saved to:", PROCESSED_DATA_PATH, "\n")

    # 3. Clustering
    print("3. Running clustering analysis...")
    elbow_method(scaled_data)

    df_clustered, silhouette = kmeans_clustering(
        df_clean,
        scaled_data,
        n_cluster=3
    )

    print("   Silhouette Score:", silhouette, "\n")
    print("Kolom dataframe:", df_clustered.columns)

    # 4. Classification
    print("4. Running classification model...")
    df_final = create_target(df_clustered)
    results = train_classification(df_final)

    print("   Accuracy:", results["accuracy"])
    print("   Confusion Matrix:\n", results["confusion_matrix"])
    print("   Classification Report:\n", results["classification_report"], "\n")

    # 5. Save Final Output
    print("5. Saving final result file...")

    # WAJIB: buat folder output/results jika belum ada
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_final.to_excel(OUTPUT_PATH, index=False)

    print("   File hasil analisis berhasil disimpan di:")
    print("   ", OUTPUT_PATH)

    print("\n=== PIPELINE FINISHED SUCCESSFULLY ===")


# =========================
# EXECUTION
# =========================

if __name__ == "__main__":
    main()
