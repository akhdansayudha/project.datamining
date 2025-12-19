import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_data(path):
    df = pd.read_excel(path)
    return df

def preprocess_data(df):
    df_clean = df.copy()

    # Encoding data kategorikal
    label_encoder = LabelEncoder()
    for col in df_clean.select_dtypes(include='object').columns:
        df_clean[col] = label_encoder.fit_transform(df_clean[col])

    # Scaling
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_clean)

    return df_clean, scaled_data

def save_processed_data(df, path):
    df.to_excel(path, index=False)
