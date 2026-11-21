
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_inquiries.csv"


def load_data():
    df = pd.read_csv(DATA_PATH)

    X = df["message"].astype(str)
    y = df["category"]

    return X, y


def train_model():

    X, y = load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2)
        )),

        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    report = classification_report(
        y_test,
        predictions,
        zero_division=0,
        output_dict=True
    )

    # Store genuine test evaluation results
    model.test_accuracy_ = accuracy
    model.test_report_ = report

    print("\nMODEL TRAINING COMPLETED")
    print("------------------------")
    print(f"Accuracy: {accuracy:.2%}")

    print("\nClassification Report:")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    return model


def predict_category(model, message):

    prediction = model.predict([message])[0]

    probabilities = model.predict_proba([message])[0]

    confidence = max(probabilities)

    return prediction, confidence


if __name__ == "__main__":

    model = train_model()

    test_message = (
        "I want an online shopping website "
        "with payment integration"
    )

    category, confidence = predict_category(
        model,
        test_message
    )

    print("\nNEW CUSTOMER INQUIRY")
    print("--------------------")
    print("Message:", test_message)
    print("Predicted Category:", category)
    print(f"Confidence: {confidence:.2%}")
