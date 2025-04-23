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
        
        text = str(row['title'])
        if pd.notna(row['abstract']) and row['abstract']:
            text += " " + str(row['abstract'])
            
       
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
        
        # 512 > BERT > 768 > Our model (768 is outut of Bert)
        D_in, H, D_out = 768, 128, NUM_CLASSES
        
        # Instantiate BERT model
        self.bert = AutoModel.from_pretrained(pretrained_model_name)
        
        # Check if the model is DistilBERT (which doesn't use token_type_ids)
        self.is_distilbert = 'distilbert' in str(type(self.bert)).lower()
        
        # Custom classification head 
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
            #No token_type_ids
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=False
            )
        else:
            # We can delete this catch now that we know it.
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

# TRAIN + EARLY STOPPING
def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, patience, max_epochs):
   
    model.train()
    best_val_f1 = 0
    no_improve_epochs = 0
    training_losses = []
    validation_metrics = []
    training_metrics = []
    
    print("\nTraining and Evaluation Metrics for Each Epoch:")
    print("-" * 80)
    print(f"{'Epoch':^6} | {'Train Loss':^10} | {'Train P':^8} | {'Train R':^8} | {'Train F1':^8} | {'Val P':^8} | {'Val R':^8} | {'Val F1':^8}")
    print("-" * 80)
    
    for epoch in range(max_epochs):
        model.train()
        total_loss = 0
        all_train_preds = []
        all_train_labels = []
        
        progress_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}/{max_epochs}')
        for batch in progress_bar:
            
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device) if 'token_type_ids' in batch else None
            labels = batch['labels'].to(device)
            
            # forward pass
            logits = model(input_ids, attention_mask, token_type_ids)
            
            # Compute loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
            
            #  backward pass to calculate gradients
            loss.backward()
            
            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()
            
            # Calculate training predictions for metrics
            probs = torch.sigmoid(logits).cpu().detach().numpy()
            preds = (probs > 0.5).astype(int)
            all_train_preds.extend(preds)
            all_train_labels.extend(labels.cpu().numpy())
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})
        
        # Calculate average training loss
        avg_train_loss = total_loss / len(train_dataloader)
        training_losses.append(avg_train_loss)
        
        # Calculate training metrics
        all_train_preds = np.array(all_train_preds)
        all_train_labels = np.array(all_train_labels)
        train_precision = precision_score(all_train_labels, all_train_preds, average='micro', zero_division=0)
        train_recall = recall_score(all_train_labels, all_train_preds, average='micro', zero_division=0)
        train_f1 = f1_score(all_train_labels, all_train_preds, average='micro', zero_division=0)
        
        # Save training metrics
        training_metrics.append({
            'precision': train_precision,
            'recall': train_recall,
            'f1': train_f1
        })
        
        # Evaluate on validation set
        val_results = evaluate_model(model, val_dataloader, device)
        val_precision = val_results['micro_avg']['precision']
        val_recall = val_results['micro_avg']['recall']
        val_f1 = val_results['micro_avg']['f1']
        
        # Save validation metrics
        validation_metrics.append({
            'precision': val_precision,
            'recall': val_recall,
            'f1': val_f1
        })
        
        # Print metrics for this epoch
        print(f"{epoch+1:^6} | {avg_train_loss:^10.4f} | {train_precision:^8.4f} | {train_recall:^8.4f} | {train_f1:^8.4f} | {val_precision:^8.4f} | {val_recall:^8.4f} | {val_f1:^8.4f}")
        
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
    
    print("-" * 80)
    
    # Plot the training metrics
    plt.figure(figsize=(15, 10))
    
    # Plot training loss
    plt.subplot(2, 2, 1)
    plt.plot(training_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    # Plot F1 scores
    plt.subplot(2, 2, 2)
    plt.plot([m['f1'] for m in training_metrics], label='Train')
    plt.plot([m['f1'] for m in validation_metrics], label='Validation')
    plt.title('F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    
    # Plot Precision
    plt.subplot(2, 2, 3)
    plt.plot([m['precision'] for m in training_metrics], label='Train')
    plt.plot([m['precision'] for m in validation_metrics], label='Validation')
    plt.title('Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    # Plot Recall
    plt.subplot(2, 2, 4)
    plt.plot([m['recall'] for m in training_metrics], label='Train')
    plt.plot([m['recall'] for m in validation_metrics], label='Validation')
    plt.title('Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Recall')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    print("Training metrics plot saved as 'training_metrics.png'")
    
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
        for batch in tqdm(dataloader, desc='Evaluating', leave=False):
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
def get_subset(dataset, fraction, random_state=42):

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
    
   
    train_subset_clinical = get_subset(train_dataset_clinical, fraction=0.05)
    val_subset_clinical = get_subset(val_dataset_clinical, fraction=0.05)
    test_subset_clinical = get_subset(test_dataset_clinical, fraction=0.2)
    
    train_subset_bio = get_subset(train_dataset_bio, fraction=0.05)
    val_subset_bio = get_subset(val_dataset_bio, fraction=0.05)
    test_subset_bio = get_subset(test_dataset_bio, fraction=0.2)
    
    print(f"Using {len(train_subset_clinical)} training samples (5% of original)")
    print(f"Using {len(val_subset_clinical)} validation samples (20% of original)")
    print(f"Using {len(test_subset_clinical)} test samples (20% of original)")
    
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
    patience = 5

    # Initialize models
    clinical_bert_model = BERTClassifier('medicalai/ClinicalBERT').to(device)
    bio_clinical_bert_model = BERTClassifier('emilyalsentzer/Bio_ClinicalBERT').to(device)
    
    Train ClinicalBERT model
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
    print("\n=== Final Model Comparison ===")
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


if __name__ == "__main__":
    main()