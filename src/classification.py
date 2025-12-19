import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def create_target(df):
    df = df.copy()

    # KLASIFIKASI LULUS (1) / TIDAK LULUS (0)
    df["Status"] = np.where(df["exam_score"] >= 75, 1, 0)

    return df


def train_classification(df):
    X = df.drop(columns=["Status"])
    y = df["Status"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred)
    }
