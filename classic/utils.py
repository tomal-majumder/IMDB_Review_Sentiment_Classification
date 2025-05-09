
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

def preprocess_data(df):
    X = df['text']
    y = df['label']
    vectorizer = TfidfVectorizer()
    X_vect = vectorizer.fit_transform(X)
    return train_test_split(X_vect, y, test_size=0.2, random_state=42)

def evaluate_model(y_true, y_pred, output_path):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    # Save predictions
    results_df = pd.DataFrame({'true': y_true, 'pred': y_pred})
    results_df.to_csv(output_path, index=False)
