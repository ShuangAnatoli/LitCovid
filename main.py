import os
import re
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, Subset
from sklearn.metrics import precision_score, recall_score, f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import random

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

MAX_LEN = 512
#Work only with the Prevention, Treatment, and Diagnosis classes.
TARGET_CLASSES = ['Prevention', 'Treatment', 'Diagnosis']
NUM_CLASSES = len(TARGET_CLASSES)

# Text preprocessing function
def text_preprocessing(text):
    """
    Clean text by removing special characters and extra whitespace
    """
    if isinstance(text, str):
        # Replace '&amp;' with '&'
        text = re.sub(r'&amp;', '&', text)
        # Remove trailing whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    return ""


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
            
        # Preprocess text
        text = text_preprocessing(text)
            
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

class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, freeze_bert=False):
        """
        @param pretrained_model_name: Name of the pretrained BERT model
        @param freeze_bert: Set to True to freeze BERT weights
        """
        super(BERTClassifier, self).__init__()
        
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 128, NUM_CLASSES
        
        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Check if the model is DistilBERT (which doesn't use token_type_ids)
        self.is_distilbert = 'distilbert' in str(type(self.bert)).lower()
        
        # Custom classification head with hidden layer
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(H, D_out)
        )
        
        # Freeze the BERT model
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
                
    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """
        Feed input to BERT and the classifier to compute logits.
        @param input_ids (torch.Tensor): an input tensor with shape (batch_size, max_length)
        @param attention_mask (torch.Tensor): a tensor that hold attention mask info
        @param token_type_ids (torch.Tensor, optional): a tensor that hold token type ids
        @return logits (torch.Tensor): an output tensor with shape (batch_size, num_labels)
        """
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

# Training function with early stopping
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, patience=3, max_epochs=10):
    """
    Train the BERTClassifier model with early stopping.
    """
    model.train()
    best_val_f1 = 0
    no_improve_epochs = 0
    training_losses = []
    validation_f1s = []
    
    for epoch in range(max_epochs):
        print(f"\nEpoch {epoch+1}/{max_epochs}")
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc='Training')
        for batch in progress_bar:
            # Zero out any previously calculated gradients
            optimizer.zero_grad()
            
            # Load batch to GPU
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            
            # Perform a forward pass
            logits = model(input_ids, attention_mask, token_type_ids)
            
            # Compute loss
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits, labels)
            
            # Perform a backward pass to calculate gradients
            loss.backward()
            
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        print(f"Training loss: {avg_train_loss:.4f}")
        
        # Evaluate on validation set
        val_results = evaluate_model(model, val_dataloader, device)
        val_f1 = val_results['micro_avg']['f1']
        validation_f1s.append(val_f1)
        print(f"Validation F1 (micro): {val_f1:.4f}")
        
        # Check if model has improved
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            no_improve_epochs = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_model.pt')
            print("Saved new best model!")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs.")
        
        # Early stopping
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
    
    # Plot the training loss and validation F1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(validation_f1s)
    plt.title('Validation F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    print("Training history plot saved as 'training_history.png'")
    
    # Load the best model
    model.load_state_dict(torch.load('best_model.pt'))
    return model, best_val_f1

# Evaluation function
def evaluate_model(model, dataloader, device):
    """
    Evaluate the BERTClassifier model.
    """
    model.eval()
    predictions = []
    true_labels = []
    
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
            
            # Convert probabilities to binary predictions (threshold = 0.5)
            preds = (probs > 0.5).astype(int)
            
            predictions.extend(preds)
            true_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(predictions)
    true_labels = np.array(true_labels)
    
    # Calculate metrics for each class
    results = {}
    for i, class_name in enumerate(TARGET_CLASSES):
        class_preds = predictions[:, i]
        class_true = true_labels[:, i]
        
        precision = precision_score(class_true, class_preds, zero_division=0)
        recall = recall_score(class_true, class_preds, zero_division=0)
        f1 = f1_score(class_true, class_preds, zero_division=0)
        
        results[class_name] = {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    # Calculate micro-averaged metrics across all classes
    micro_precision = precision_score(true_labels, predictions, average='micro', zero_division=0)
    micro_recall = recall_score(true_labels, predictions, average='micro', zero_division=0)
    micro_f1 = f1_score(true_labels, predictions, average='micro', zero_division=0)
    
    results['micro_avg'] = {
        'precision': micro_precision,
        'recall': micro_recall,
        'f1': micro_f1
    }
    
    return results

# Function to create dataloader
def create_dataloader(dataset, batch_size, is_train=True):
    if is_train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
    
    return DataLoader(dataset, sampler=sampler, batch_size=batch_size)

# Function to get a subset of a dataset
def get_subset(dataset, fraction=0.1, random_state=42):
    """
    Get a random subset of a dataset.
    
    Args:
        dataset: A pytorch Dataset object
        fraction: The fraction of data to use (0.1 = 10%)
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

# Main function
def main():
    # Load data
    train_data = pd.read_csv('BC7-LitCovid-Train.csv')
    test_data = pd.read_csv('BC7-LitCovid-Test-GS.csv')
    
    print(f"Training data shape: {train_data.shape}")
    print(f"Test data shape: {test_data.shape}")
    
    def filter_classes(df):
        filtered_rows = []
        for idx, row in df.iterrows():
            if pd.notna(row['label']):
                labels = row['label'].split(';')
                if any(label in TARGET_CLASSES for label in labels):
                    filtered_rows.append(row)
        return pd.DataFrame(filtered_rows)
    
    train_data = filter_classes(train_data)
    test_data = filter_classes(test_data)
    
    print(f"Filtered training data shape: {train_data.shape}")
    print(f"Filtered test data shape: {test_data.shape}")
    
    # Split training data into train and validation sets (80/20)
    train_size = int(0.8 * len(train_data))
    train_data_split = train_data.iloc[:train_size]
    val_data = train_data.iloc[train_size:]
    
    print(f"Train split size: {len(train_data_split)}")
    print(f"Validation split size: {len(val_data)}")
    
    # Initialize tokenizers
    clinical_bert_tokenizer = AutoTokenizer.from_pretrained('medicalai/ClinicalBERT')
    bio_clinical_bert_tokenizer = AutoTokenizer.from_pretrained('emilyalsentzer/Bio_ClinicalBERT')
    
    # Create datasets
    train_dataset_clinical = LitCovidDataset(train_data_split, clinical_bert_tokenizer)
    val_dataset_clinical = LitCovidDataset(val_data, clinical_bert_tokenizer)
    test_dataset_clinical = LitCovidDataset(test_data, clinical_bert_tokenizer)
    
    train_dataset_bio = LitCovidDataset(train_data_split, bio_clinical_bert_tokenizer)
    val_dataset_bio = LitCovidDataset(val_data, bio_clinical_bert_tokenizer)
    test_dataset_bio = LitCovidDataset(test_data, bio_clinical_bert_tokenizer)
    
    # Take 10% of the datasets
    train_subset_clinical = get_subset(train_dataset_clinical, fraction=0.1)
    val_subset_clinical = get_subset(val_dataset_clinical, fraction=0.1)
    test_subset_clinical = get_subset(test_dataset_clinical, fraction=0.1)
    
    train_subset_bio = get_subset(train_dataset_bio, fraction=0.1)
    val_subset_bio = get_subset(val_dataset_bio, fraction=0.1)
    test_subset_bio = get_subset(test_dataset_bio, fraction=0.1)
    
    print(f"Using {len(train_subset_clinical)} training samples (10% of original)")
    print(f"Using {len(val_subset_clinical)} validation samples (10% of original)")
    print(f"Using {len(test_subset_clinical)} test samples (10% of original)")
    
    # Create dataloaders
    batch_size = 16
    train_dataloader_clinical = create_dataloader(train_subset_clinical, batch_size, is_train=True)
    val_dataloader_clinical = create_dataloader(val_subset_clinical, batch_size, is_train=False)
    test_dataloader_clinical = create_dataloader(test_subset_clinical, batch_size, is_train=False)
    
    train_dataloader_bio = create_dataloader(train_subset_bio, batch_size, is_train=True)
    val_dataloader_bio = create_dataloader(val_subset_bio, batch_size, is_train=False)
    test_dataloader_bio = create_dataloader(test_subset_bio, batch_size, is_train=False)
    
    # Training parameters
    learning_rate = 2e-5
    max_epochs = 100
    patience = 2 
    
    # Initialize models
    clinical_bert_model = BERTClassifier('medicalai/ClinicalBERT').to(device)
    bio_clinical_bert_model = BERTClassifier('emilyalsentzer/Bio_ClinicalBERT').to(device)
    
    # Train ClinicalBERT model
    print("\n=== Training ClinicalBERT Model ===")
    optimizer_clinical = AdamW(clinical_bert_model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader_clinical) * max_epochs
    scheduler_clinical = get_linear_schedule_with_warmup(
        optimizer_clinical,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    clinical_bert_model, best_val_f1_clinical = train_model(
        clinical_bert_model,
        train_dataloader_clinical,
        val_dataloader_clinical,
        optimizer_clinical,
        scheduler_clinical,
        device,
        patience=patience,
        max_epochs=max_epochs
    )
    
    # Evaluate ClinicalBERT model
    print("\n=== Evaluating ClinicalBERT Model ===")
    clinical_bert_results = evaluate_model(clinical_bert_model, test_dataloader_clinical, device)
    
    # Train Bio_ClinicalBERT model
    print("\n=== Training Bio_ClinicalBERT Model ===")
    optimizer_bio = AdamW(bio_clinical_bert_model.parameters(), lr=learning_rate)
    total_steps = len(train_dataloader_bio) * max_epochs
    scheduler_bio = get_linear_schedule_with_warmup(
        optimizer_bio,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    bio_clinical_bert_model, best_val_f1_bio = train_model(
        bio_clinical_bert_model,
        train_dataloader_bio,
        val_dataloader_bio,
        optimizer_bio,
        scheduler_bio,
        device,
        patience=patience,
        max_epochs=max_epochs
    )
    
    # Evaluate Bio_ClinicalBERT model
    print("\n=== Evaluating Bio_ClinicalBERT Model ===")
    bio_clinical_bert_results = evaluate_model(bio_clinical_bert_model, test_dataloader_bio, device)
    
    # Print results
    print("\n=== Model Comparison ===")
    print("\nClinicalBERT Results:")
    for class_name in TARGET_CLASSES:
        metrics = clinical_bert_results[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    print("\nBio_ClinicalBERT Results:")
    for class_name in TARGET_CLASSES:
        metrics = bio_clinical_bert_results[class_name]
        print(f"{class_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
    
    # Print micro-average results
    print("\nMicro-Average Results:")
    print("ClinicalBERT:")
    print(f"  Precision: {clinical_bert_results['micro_avg']['precision']:.4f}")
    print(f"  Recall: {clinical_bert_results['micro_avg']['recall']:.4f}")
    print(f"  F1 Score: {clinical_bert_results['micro_avg']['f1']:.4f}")
    
    print("Bio_ClinicalBERT:")
    print(f"  Precision: {bio_clinical_bert_results['micro_avg']['precision']:.4f}")
    print(f"  Recall: {bio_clinical_bert_results['micro_avg']['recall']:.4f}")
    print(f"  F1 Score: {bio_clinical_bert_results['micro_avg']['f1']:.4f}")
    
    # Create visualization to compare models
    create_comparison_plot(clinical_bert_results, bio_clinical_bert_results)

def create_comparison_plot(clinical_bert_results, bio_clinical_bert_results):
    """Create bar plots to compare model performance"""
    metrics = ['precision', 'recall', 'f1']
    fig, axes = plt.subplots(len(TARGET_CLASSES), 3, figsize=(15, 12))
    
    for i, class_name in enumerate(TARGET_CLASSES):
        for j, metric in enumerate(metrics):
            clinical_val = clinical_bert_results[class_name][metric]
            bio_val = bio_clinical_bert_results[class_name][metric]
            
            ax = axes[i, j]
            bars = ax.bar(['ClinicalBERT', 'Bio_ClinicalBERT'], [clinical_val, bio_val])
            ax.set_ylim(0, 1.0)
            ax.set_title(f"{class_name} - {metric.capitalize()}")
            
            # Add values on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    print("\nComparison plot saved as 'model_comparison.png'")

if __name__ == "__main__":
    main()