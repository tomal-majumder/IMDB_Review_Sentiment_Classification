
import argparse
import pandas as pd
import re
import nltk
import string
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

def main():
    parser = argparse.ArgumentParser(description="Preprocess a dataset for sentiment classification")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV with 'text' column")
    parser.add_argument("--output", type=str, default="cleaned_dataset.csv", help="Path to save cleaned CSV")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df['text'] = df['text'].apply(clean_text)
    df.to_csv(args.output, index=False)
    print(f"Cleaned dataset saved to {args.output}")

if __name__ == "__main__":
    main()
