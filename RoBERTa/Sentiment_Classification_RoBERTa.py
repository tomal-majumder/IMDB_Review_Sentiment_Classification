
import pandas as pd

# Load the preprocessed data
processed_df = pd.read_csv('/content/drive/MyDrive/Sentiment_Data/processed_reviews.csv')

# Now you can work with the 'processed_df' DataFrame
print(processed_df.head())


processed_df.head()

import re
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, RobertaTokenizer, RobertaConfig
import transformers
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

config = AutoConfig.from_pretrained("roberta-base")
MAX_LEN = 512
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Dataset Analysis Functions
def analyze_dataset_balance(df, label_column="label"):
    """Analyze if the dataset is balanced and print statistics."""
    label_counts = df[label_column].value_counts()
    total = len(df)

    print("\n=== Dataset Balance Analysis ===")
    for label, count in label_counts.items():
        print(f"Label {label}: {count} samples ({count/total*100:.2f}%)")

    # Calculate imbalance ratio
    if len(label_counts) == 2:
        imbalance_ratio = label_counts.max() / label_counts.min()
        print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

        if imbalance_ratio > 2:
            print("WARNING: Dataset is imbalanced (ratio > 2:1)")
            print("Consider using weighted loss function or sampling techniques")

    return label_counts

def analyze_text_length(df, text_column="processed_review"):
    """Analyze text length distribution."""
    df['text_length'] = df[text_column].apply(lambda x: len(str(x).split()))

    print("\n=== Text Length Analysis ===")
    print(f"Mean length: {df['text_length'].mean():.2f} words")
    print(f"Median length: {df['text_length'].median()} words")
    print(f"Max length: {df['text_length'].max()} words")

    # Check if we might be truncating too much
    long_texts = (df['text_length'] > 400).sum()
    print(f"Texts longer than 400 words: {long_texts} ({long_texts/len(df)*100:.2f}%)")

    return df['text_length'].describe()


class CustomDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.comment_text = dataframe["processed_review"].tolist()
        self.targets = dataframe["label"].tolist()
        self.max_len = max_len

    def __len__(self):
        return len(self.comment_text)

    def __getitem__(self, index):
        comment_text = str(self.comment_text[index])
        comment_text = " ".join(comment_text.split())
        inputs = self.tokenizer.encode_plus(
            comment_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_token_type_ids=True,
            truncation=True,
            return_tensors='pt'
        )
        ids = inputs['input_ids'].squeeze()
        mask = inputs['attention_mask'].squeeze()
        token_type_ids = inputs["token_type_ids"].squeeze()

        return {
            'ids': ids,
            'mask': mask,
            'token_type_ids': token_type_ids,
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

# Creating the customized model, by adding a drop out and a dense layer on top of distil bert to get the final output for the model.
class BERTClass(torch.nn.Module):
    def __init__(self):
        super(BERTClass, self).__init__()
        self.l1 = transformers.RobertaModel.from_pretrained("roberta-base")
        self.l2 = torch.nn.Dropout(0.2)
        self.l3 = torch.nn.Linear(768, 1)

    def forward(self, ids, mask, token_type_ids):
        # Pass attention_mask explicitly to address the warning
        output_1 = self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids)
        output_2 = self.l2(output_1.pooler_output)  # Using the pooler_output
        output = self.l3(output_2)
        return output

def evaluate(model, dataloader):
    """Evaluate the model on the given dataloader."""
    model.eval()
    predictions = []
    actual_labels = []

    with torch.no_grad():
        for data in tqdm(dataloader, desc="Evaluating"):
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].cpu().numpy()

            outputs = model(ids, mask, token_type_ids)
            outputs = torch.sigmoid(outputs.squeeze()).cpu().numpy()

            predictions.extend((outputs > 0.5).astype(int))
            actual_labels.extend(targets.astype(int))

    # Calculate metrics
    accuracy = accuracy_score(actual_labels, predictions)
    f1 = f1_score(actual_labels, predictions)
    precision = precision_score(actual_labels, predictions)
    recall = recall_score(actual_labels, predictions)

    print("\n=== Evaluation Results ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# Main execution
if __name__ == "__main__":
    # Ensure processed_df is available
    # This is a placeholder - replace with your actual data loading
    try:
        print("Working with existing processed_df...")
        # Analyze dataset balance
        analyze_dataset_balance(processed_df)
        analyze_text_length(processed_df)
    except NameError:
        print("Warning: processed_df not found. Replace this with your actual data loading code.")
        # If you need to load the dataset here, uncomment and modify the line below:
        # processed_df = pd.read_csv('your_data.csv')

    # Your original train/test split
    train_size = 0.7
    train_dataset = processed_df.sample(frac=train_size, random_state=42)
    test_dataset = processed_df.drop(train_dataset.index).reset_index(drop=True)
    train_dataset = train_dataset.reset_index(drop=True)

    print("FULL Dataset: {}".format(processed_df.shape))
    print("TRAIN Dataset: {}".format(train_dataset.shape))
    print("TEST Dataset: {}".format(test_dataset.shape))

    # Create datasets
    training_set = CustomDataset(train_dataset, tokenizer, MAX_LEN)
    testing_set = CustomDataset(test_dataset, tokenizer, MAX_LEN)

    # Increased batch size and optimized dataloader parameters
    TRAIN_BATCH_SIZE = 32  # Increased from 5 to 32
    VALID_BATCH_SIZE = 32  # Increased from 1 to 32

    train_params = {
        'batch_size': TRAIN_BATCH_SIZE,
        'shuffle': True,
        'num_workers': 4,  # Increased from 0 to 4
        'pin_memory': True  # Added for faster data transfer to GPU
    }

    test_params = {
        'batch_size': VALID_BATCH_SIZE,
        'shuffle': False,  # No need to shuffle test data
        'num_workers': 4,
        'pin_memory': True
    }

    train_dataloader = DataLoader(training_set, **train_params)
    testing_loader = DataLoader(testing_set, **test_params)

    # Initialize model
    model = BERTClass()
    model.to(device)

    # Gradient accumulation steps
    ACCUMULATION_STEPS = 2  # Effective batch size = 32 * 2 = 64

    # Improved optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

    # Learning rate scheduler
    total_steps = len(train_dataloader) * 3 // ACCUMULATION_STEPS  # 3 epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% warmup
        num_training_steps=total_steps
    )

    # Loss function
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Initialize mixed precision training
    scaler = GradScaler()

    # Training loop
    EPOCHS = 3
    best_f1 = 0

    # Calculate approximate time
    steps_per_epoch = len(train_dataloader)
    updates_per_epoch = steps_per_epoch // ACCUMULATION_STEPS
    total_updates = updates_per_epoch * EPOCHS

    print(f"\nTraining Configuration:")
    print(f"- Train samples: {len(train_dataset)}")
    print(f"- Batch size: {TRAIN_BATCH_SIZE}")
    print(f"- Gradient accumulation steps: {ACCUMULATION_STEPS}")
    print(f"- Effective batch size: {TRAIN_BATCH_SIZE * ACCUMULATION_STEPS}")
    print(f"- Steps per epoch: {steps_per_epoch}")
    print(f"- Optimizer updates per epoch: {updates_per_epoch}")
    print(f"- Total optimizer updates for {EPOCHS} epochs: {total_updates}")
    print(f"- Using mixed precision: Yes")

    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"EPOCH {epoch+1}/{EPOCHS}")
        print(f"{'='*50}")

        # Training
        model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),
                           desc=f"Epoch {epoch+1} Training")

        for i, data in progress_bar:
            ids = data['ids'].to(device, dtype=torch.long)
            mask = data['mask'].to(device, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype=torch.long)
            targets = data['targets'].to(device, dtype=torch.float)

            # Mixed precision training
            with autocast():
                outputs = model(ids, mask, token_type_ids)
                loss = loss_fn(outputs.squeeze(), targets)
                loss = loss / ACCUMULATION_STEPS  # Normalize loss for accumulation

            # Scale loss and accumulate gradients
            scaler.scale(loss).backward()

            # Update only after accumulation steps
            if (i + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * ACCUMULATION_STEPS
            progress_bar.set_postfix({'loss': total_loss / (i+1)})

        # Make sure to update for the last batch if dataset size is not a multiple of ACCUMULATION_STEPS
        if len(train_dataloader) % ACCUMULATION_STEPS != 0:
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        avg_train_loss = total_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss:.4f}")

        # Evaluation
        metrics = evaluate(model, testing_loader)

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "/content/drive/MyDrive/Sentiment_Data/best_model.pt")
            print(f"New best model saved with F1: {best_f1:.4f}")

    print("\nTraining completed!")
    print(f"Best F1 Score: {best_f1:.4f}")



    # Load best model and final evaluation
    model.load_state_dict(torch.load("/content/drive/MyDrive/Sentiment_Data/best_model.pt"))
    print("\nFinal evaluation with best model:")
    final_metrics = evaluate(model, testing_loader)