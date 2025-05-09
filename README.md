
# Sentiment Classification: Classic Models vs fine-tuned RoBERTa

This repository compares traditional machine learning classifiers and a transformer-based RoBERTa model for sentiment classification. It includes:

- Data cleaning and preprocessing pipeline
- Classic classifiers: Naive Bayes, SVM, Logistic Regression
- RoBERTa-based classifier (fine-tuned with HuggingFace Transformers)

---

## Project Structure

```
sentiment-classifier/
├── preprocessing.py 
├── classic/
│   ├── nb_classifier.py           # Naive Bayes model
│   ├── svm_classifier.py          # Support Vector Machine model
│   ├── lr_classifier.py           # Logistic Regression model
│   └── utils.py                   # Shared functions for evaluation
│
├── roberta/
│   └── Sentiment_Classification_RoBERTa.py
├── requirements.txt
└── README.md
```

---

## Dataset Format

The input CSV must contain:

```csv
text,label
"I love this!",1
"This is awful",0
```

---

## Installation

Install all dependencies for both classic and transformer models:

```bash
pip install -r requirements.txt
```

---

## Step 1: Preprocess Dataset

```bash
python preprocessing.py --input data/your_dataset.csv --output data/cleaned_dataset.csv
```

- `--input`: Path to raw dataset with `text` and `label` columns.
- `--output`: Cleaned dataset with normalized text and stopword removal.

---

## Step 2: Run Classic Classifiers

Run one of the classic models:

### Naive Bayes

```bash
python classic/nb_classifier.py --input data/cleaned_dataset.csv --output nb_results.csv
```

### SVM

```bash
python classic/svm_classifier.py --input data/cleaned_dataset.csv --output svm_results.csv
```

### Logistic Regression

```bash
python classic/lr_classifier.py --input data/cleaned_dataset.csv --output lr_results.csv
```

---

## 🤖 Step 3: Run RoBERTa Classifier

```bash
python roberta/Sentiment_Classification_RoBERTa.py
```

> ⚠️ This script assumes dataset is loaded inside the script. You can modify it to accept command-line arguments like the classic ones.

---

## Output

Each classifier prints:

- Accuracy score
- Classification report
- Confusion matrix

Results (true vs predicted labels) are saved in the CSV specified via `--output`.

---

## Requirements

See `requirements.txt` for the full list.

---

## Interactive Run
If you'd like to explore and visualize the models step-by-step, run the following notebooks:

Sentiment_Classification_classic.ipynb – for training and comparing classic ML models (SVM, NB, LR)

Sentiment_Classification_RoBERTa.ipynb – for fine-tuning and evaluating a RoBERTa model on sentiment data

>💡 We recommend running these in JupyterLab or Google Colab for best experience.

---
