import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SequentialSampler, Subset
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import random

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Constants
MAX_LEN = 512
TARGET_CLASSES = ['Prevention', 'Treatment', 'Diagnosis']
NUM_CLASSES = len(TARGET_CLASSES)

# Dataset class
class LitCovidDataset(Dataset):
    def __init__(self, data, tokenizer, max_len=MAX_LEN):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        row = self.data.iloc[index]
        
        # Combine title and abstract
        text = str(row['title'])
        if pd.notna(row['abstract']) and row['abstract']:
            text += " " + str(row['abstract'])
            
        # Tokenize text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        labels = torch.zeros(NUM_CLASSES)
        if pd.notna(row['label']):
            label_list = str(row['label']).split(';')
            for label in label_list:
                if label in TARGET_CLASSES:
                    labels[TARGET_CLASSES.index(label)] = 1.0
                    
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding['token_type_ids'].flatten() if 'token_type_ids' in encoding else None,
            'labels': labels
        }

# Model class
class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, freeze_bert=False):
        super(BERTClassifier, self).__init__()
        
        # Specify hidden size of BERT, hidden size of classifier, and number of labels
        D_in, H, D_out = 768, 128, NUM_CLASSES
        
        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Check if the model is DistilBERT
        self.is_distilbert = 'distilbert' in str(type(self.bert)).lower()
        
        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(H, D_out)
        )
                
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # Feed input to BERT
        if self.is_distilbert:
            # DistilBERT doesn't use token_type_ids
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            # Regular BERT uses token_type_ids
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_attentions=False,
                output_hidden_states=False
            )
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]
        
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)
        
        return logits

def evaluate_model(model, dataloader, device):
    """
    Evaluate the model and generate metrics
    """
    model.eval()
    all_predictions = []
    all_true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels']
            
            # Compute logits
            logits = model(input_ids, attention_mask, token_type_ids)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            
            # Convert probabilities to binary predictions
            preds = (probs > 0.5).astype(int)
            
            all_predictions.extend(preds)
            all_true_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_true_labels = np.array(all_true_labels)
    
    # Calculate metrics for each class
    results = {}
    for i, class_name in enumerate(TARGET_CLASSES):
        class_preds = all_predictions[:, i]
        class_true = all_true_labels[:, i]
        
        precision = precision_score(class_true, class_preds, zero_division=0)
        recall = recall_score(class_true, class_preds, zero_division=0)
        f1 = f1_score(class_true, class_preds, zero_division=0)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate micro-averaged metrics across all classes
    micro_precision = precision_score(all_true_labels, all_predictions, average='micro', zero_division=0)
    micro_recall = recall_score(all_true_labels, all_predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(all_true_labels, all_predictions, average='micro', zero_division=0)
    
    results['micro_avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1
    }
    
    # Calculate macro-averaged metrics
    macro_precision = precision_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    macro_recall = recall_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    macro_f1 = f1_score(all_true_labels, all_predictions, average='macro', zero_division=0)
    
    results['macro_avg'] = {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }
    
    # Calculate weighted-averaged metrics
    weighted_precision = precision_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    weighted_recall = recall_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    weighted_f1 = f1_score(all_true_labels, all_predictions, average='weighted', zero_division=0)
    
    results['weighted_avg'] = {
        'precision': weighted_precision,
        'recall': weighted_recall,
        'f1': weighted_f1
    }
    
    # Fix for the classification report - use sample-based approach
    # Create lists to store sample indices and their corresponding true/predicted labels
    sample_idx_to_true_labels = {}
    sample_idx_to_pred_labels = {}
    
    # Process each sample and collect all its true and predicted labels
    for i in range(len(all_true_labels)):
        true_labels_for_sample = []
        pred_labels_for_sample = []
        
        for j in range(NUM_CLASSES):
            if all_true_labels[i, j] == 1:
                true_labels_for_sample.append(TARGET_CLASSES[j])
            if all_predictions[i, j] == 1:
                pred_labels_for_sample.append(TARGET_CLASSES[j])
        
        # Ensure at least one label per sample (use 'None' if no labels)
        if not true_labels_for_sample:
            true_labels_for_sample = ['None']
        if not pred_labels_for_sample:
            pred_labels_for_sample = ['None']
            
        sample_idx_to_true_labels[i] = true_labels_for_sample
        sample_idx_to_pred_labels[i] = pred_labels_for_sample
    
    # Now create balanced flat lists for classification report
    y_true_flat = []
    y_pred_flat = []
    
    # Create balanced pairs of true and predicted labels
    for i in range(len(all_true_labels)):
        true_labels = sample_idx_to_true_labels[i]
        pred_labels = sample_idx_to_pred_labels[i]
        
        # Case 1: Same number of true and predicted labels - direct pairing
        if len(true_labels) == len(pred_labels):
            for j in range(len(true_labels)):
                y_true_flat.append(true_labels[j])
                y_pred_flat.append(pred_labels[j])
        
        # Case 2: More true labels than predicted - pad predictions with 'None'
        elif len(true_labels) > len(pred_labels):
            # Add all paired labels first
            for j in range(len(pred_labels)):
                y_true_flat.append(true_labels[j])
                y_pred_flat.append(pred_labels[j])
            
            # Add remaining true labels with 'None' predictions
            for j in range(len(pred_labels), len(true_labels)):
                y_true_flat.append(true_labels[j])
                y_pred_flat.append('None')
        
        # Case 3: More predicted labels than true - pad true labels with 'None'
        else:
            # Add all paired labels first
            for j in range(len(true_labels)):
                y_true_flat.append(true_labels[j])
                y_pred_flat.append(pred_labels[j])
            
            # Add remaining predicted labels with 'None' true labels
            for j in range(len(true_labels), len(pred_labels)):
                y_true_flat.append('None')
                y_pred_flat.append(pred_labels[j])
    
    # Verify that we have the same number of elements
    assert len(y_true_flat) == len(y_pred_flat), "Mismatch in sample counts after balancing"
    
    # Add 'None' to the target classes for the classification report
    report_labels = TARGET_CLASSES.copy()
    if 'None' in y_true_flat or 'None' in y_pred_flat:
        report_labels.append('None')
    
    # Generate classification report (now with balanced lists)
    cr = classification_report(
        y_true_flat, y_pred_flat, 
        labels=report_labels, 
        zero_division=0,
        output_dict=True
    )
    
    results['classification_report'] = cr
    
    return results, all_predictions, all_true_labels, y_true_flat, y_pred_flat

def get_subset(dataset, fraction=0.2, random_state=42):
    """
    Get a random subset of a dataset.
    
    Args:
        dataset: A pytorch Dataset object
        fraction: The fraction of data to use (0.2 = 20%)
        random_state: Seed for reproducibility
        
    Returns:
        Subset of the dataset
    """
    total_size = len(dataset)
    subset_size = int(total_size * fraction)
    
    # Set random seed for reproducibility
    random.seed(random_state)
    indices = random.sample(range(total_size), subset_size)
    
    return Subset(dataset, indices)

def main():
    """
    Load model, evaluate it on 20% of test data, and generate a classification report
    """
    # Load test data
    test_data = pd.read_csv('BC7-LitCovid-Test-GS.csv')
    print(f"Test data shape: {test_data.shape}")
    
    # Filter for target classes
    def filter_classes(df):
        filtered_rows = []
        for idx, row in df.iterrows():
            if pd.notna(row['label']):
                labels = row['label'].split(';')
                if any(label in TARGET_CLASSES for label in labels):
                    filtered_rows.append(row)
        return pd.DataFrame(filtered_rows)
    
    filtered_test_data = filter_classes(test_data)
    print(f"Filtered test data shape: {filtered_test_data.shape}")
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Create full dataset
    full_test_dataset = LitCovidDataset(filtered_test_data, tokenizer)
    
    # Create 20% subset
    test_subset = get_subset(full_test_dataset, fraction=0.2)
    print(f"Using {len(test_subset)} samples (20% of filtered test data)")
    
    # Create dataloader for the subset
    test_dataloader = DataLoader(test_subset, batch_size=16, shuffle=False)
    
    # Initialize model
    model = BERTClassifier('emilyalsentzer/Bio_ClinicalBERT').to(device)
    
    # Load the saved model weights
    try:
        model.load_state_dict(torch.load('best_model.pt', map_location=device))
        print("Successfully loaded model from 'best_model.pt'")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Evaluate model
    print("Evaluating model on test data...")
    results, predictions, true_labels, y_true_flat, y_pred_flat = evaluate_model(model, test_dataloader, device)
    
    # Print results
    print("\n===== Model Evaluation Results =====")
    print("\nPer-class metrics:")
    for class_name in TARGET_CLASSES:
        metrics = results[class_name]
        print(f"\n{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print("\nAverage metrics:")
    print("Micro Average:")
    print(f"  Precision: {results['micro_avg']['precision']:.4f}")
    print(f"  Recall: {results['micro_avg']['recall']:.4f}")
    print(f"  F1 Score: {results['micro_avg']['f1']:.4f}")
    
    print("\nMacro Average:")
    print(f"  Precision: {results['macro_avg']['precision']:.4f}")
    print(f"  Recall: {results['macro_avg']['recall']:.4f}")
    print(f"  F1 Score: {results['macro_avg']['f1']:.4f}")
    
    print("\nWeighted Average:")
    print(f"  Precision: {results['weighted_avg']['precision']:.4f}")
    print(f"  Recall: {results['weighted_avg']['recall']:.4f}")
    print(f"  F1 Score: {results['weighted_avg']['f1']:.4f}")
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true_flat, y_pred_flat, labels=TARGET_CLASSES, zero_division=0))
    
    # Save the results to a file
    with open('evaluation_results_bio_clinical.txt', 'w') as f:
        f.write("===== Model Evaluation Results (20% of Test Data) =====\n")
        
        f.write("\nPer-class metrics:\n")
        for class_name in TARGET_CLASSES:
            metrics = results[class_name]
            f.write(f"\n{class_name}:\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1']:.4f}\n")
        
        f.write("\nAverage metrics:\n")
        f.write("Micro Average:\n")
        f.write(f"  Precision: {results['micro_avg']['precision']:.4f}\n")
        f.write(f"  Recall: {results['micro_avg']['recall']:.4f}\n")
        f.write(f"  F1 Score: {results['micro_avg']['f1']:.4f}\n")
        
        f.write("\nMacro Average:\n")
        f.write(f"  Precision: {results['macro_avg']['precision']:.4f}\n")
        f.write(f"  Recall: {results['macro_avg']['recall']:.4f}\n")
        f.write(f"  F1 Score: {results['macro_avg']['f1']:.4f}\n")
        
        f.write("\nWeighted Average:\n")
        f.write(f"  Precision: {results['weighted_avg']['precision']:.4f}\n")
        f.write(f"  Recall: {results['weighted_avg']['recall']:.4f}\n")
        f.write(f"  F1 Score: {results['weighted_avg']['f1']:.4f}\n")
        
        f.write("\nClassification Report:\n")
        f.write(classification_report(y_true_flat, y_pred_flat, labels=TARGET_CLASSES, zero_division=0))
    
    print("\nEvaluation results saved to 'evaluation_results_bio_clinical.txt'")

if __name__ == "__main__":
    main()