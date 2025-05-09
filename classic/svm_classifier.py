
import argparse
import pandas as pd
from utils import preprocess_data, evaluate_model

def main():
    parser = argparse.ArgumentParser(description="Run SVM sentiment classifier")
    parser.add_argument("--input", type=str, required=True, help="Path to input CSV with 'text' and 'label' columns")
    parser.add_argument("--output", type=str, default="svm_results.csv", help="Path to save output CSV with predictions")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    X_train, X_test, y_train, y_test = preprocess_data(df)


    from sklearn.svm import LinearSVC
    model = LinearSVC()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    evaluate_model(y_test, predictions, args.output)

if __name__ == "__main__":
    main()
