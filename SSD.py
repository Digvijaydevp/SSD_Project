#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from nltk.corpus import wordnet
from itertools import chain
import re
import random
import pandas as pd
import numpy as np


# In[2]:


# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Clean tweet function (remains the same)
def clean_tweet(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text


# In[3]:


# Synonym replacement and augmentation function
def get_synonyms(word):
    synonyms = wordnet.synsets(word)
    lemmas = set(chain.from_iterable([syn.lemma_names() for syn in synonyms]))
    return lemmas


# In[4]:


def synonym_replacement(sentence, n=1):
    words = sentence.split()
    random.shuffle(words)
    for _ in range(n):
        for word in words:
            synonyms = get_synonyms(word)
            if synonyms:
                sentence = sentence.replace(word, random.choice(list(synonyms)), 1)
                break
    return sentence


# In[5]:


# Load and preprocess data
file_path = './Tweets.xlsx'
df = pd.read_excel(file_path)
df['text'] = df['text'].apply(clean_tweet)


# In[6]:


# Map sentiments to numerical labels
sentiment_mapping = {'positive': 2, 'neutral': 1, 'negative': 0}
df['label'] = df['sentiment'].map(sentiment_mapping)


# In[7]:


# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# In[10]:


class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len, augment=False):
        self.texts = texts
        self.original_texts = texts  # Store original unprocessed texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.augment = augment

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        original_text = self.original_texts[index]  # Get the original text
        label = self.labels[index]

        # Apply sarcasm-specific augmentation (synonym replacement)
        if self.augment and random.uniform(0, 1) > 0.5:
            text = synonym_replacement(text)

        # Tokenize the text
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'text': text,  # The augmented/processed text
            'original_text': original_text,  # The original unprocessed text
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


# Perform train-validation-test split
# First split into train+validation and test sets
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Then split train+validation into train and validation sets
train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df['label'], random_state=42)



# In[11]:


# Prepare datasets
MAX_LEN = 80
BATCH_SIZE = 8

train_dataset = TweetDataset(train_df['text'].tolist(), train_df['label'].tolist(), tokenizer, MAX_LEN, augment=True)
val_dataset = TweetDataset(val_df['text'].tolist(), val_df['label'].tolist(), tokenizer, MAX_LEN, augment=False)
test_dataset = TweetDataset(test_df['text'].tolist(), test_df['label'].tolist(), tokenizer, MAX_LEN, augment=False)


# In[12]:


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Model setup
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.to(device)

# Calculate class weights
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)


# In[13]:


# Import necessary modules for training
from torch.cuda.amp import GradScaler, autocast

# Initialize the GradScaler for mixed precision
scaler = GradScaler()

# Training functions
def train_epoch(model, data_loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    total_batches = len(data_loader)
    
    for batch_index, batch in enumerate(data_loader):
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # Print progress
        print(f'Epoch {epoch + 1}, Batch {batch_index + 1}/{total_batches} completed.')

    # Prevent division by zero
    avg_loss = total_loss / max(total_batches, 1)
    return avg_loss


# In[14]:


def eval_model(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += torch.sum(preds == labels)

    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def save_test_predictions(model, test_loader, device, output_file="test_predictions.csv"):
    model.eval()
    test_results = []

    # Mapping from numeric labels to sentiment labels
    label_to_sentiment = {0: 'negative', 1: 'neutral', 2: 'positive'}

    with torch.no_grad():
        for batch_index, batch in enumerate(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            # Get model outputs
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get predicted labels
            _, preds = torch.max(logits, dim=1)


            for i in range(len(batch['input_ids'])):
                test_results.append({
                    'text': batch['original_text'][i],  
                    'true_label': label_to_sentiment[batch['label'][i].item()],  
                    'predicted_label': label_to_sentiment[preds[i].item()]      
                })

            print(f'Processed batch {batch_index + 1}/{len(test_loader)}.')

    # Save results to CSV
    df = pd.DataFrame(test_results)
    df.to_csv(output_file, index=False)
    print(f'Test predictions saved to {output_file}')



# In[18]:


# Create lists to track metrics
training_history = []
performance_metrics = {}

# Training loop with updated parameters
EPOCHS = 3
for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    
    # Train the model and get train loss
    train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    
    # Evaluate on validation set
    val_loss, val_accuracy = eval_model(model, val_loader, criterion, device)
    
    # Store metrics for each epoch
    training_history.append({
        'epoch': epoch + 1,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'val_accuracy': val_accuracy
    })
    
    print(f'Train Loss: {train_loss}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}')

# Calculate additional performance metrics
def calculate_performance_metrics(model, test_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            # Extract inputs and labels from the batch
            inputs = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)  # Add attention mask
            labels = batch['label'].to(device)

            # Perform forward pass and extract logits
            outputs = model(inputs, attention_mask=attention_mask)
            logits = outputs.logits  

            # Get the predicted class
            _, predicted = torch.max(logits, 1)

            # Collect predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate metrics
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )

    performance_metrics = {
        'accuracy': accuracy_score(all_labels, all_preds),
        'precision': precision_score(all_labels, all_preds, average='weighted'),
        'recall': recall_score(all_labels, all_preds, average='weighted'),
        'f1_score': f1_score(all_labels, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_preds).tolist(),
    }

    # Calculate class distribution
    unique, counts = np.unique(all_preds, return_counts=True)
    class_distribution = [
        {'name': f'Class {c}', 'value': count}
        for c, count in zip(unique, counts)
    ]

    return performance_metrics, class_distribution


performance_metrics, class_distribution = calculate_performance_metrics(model, test_loader, device)


import json
import numpy as np

# Save performance metrics
with open('performance_metrics.json', 'w') as f:
    json.dump({k: float(v) if isinstance(v, np.float64) else v for k, v in performance_metrics.items()}, f)

# Convert class_distribution to ensure JSON serializability
class_distribution = [
    {'name': item['name'], 'value': int(item['value'])}
    for item in class_distribution
]

# Save class distribution
with open('class_distribution.json', 'w') as f:
    json.dump(class_distribution, f)

# Convert val_accuracy tensors to floats
for record in training_history:
    if isinstance(record['val_accuracy'], torch.Tensor):
        record['val_accuracy'] = record['val_accuracy'].item()  

# Save training history
with open('training_history.json', 'w') as f:
    json.dump(training_history, f)


# In[ ]:


# Prediction saving after fine-tuning
save_test_predictions(model, test_loader, device, output_file="test_predictions_after_finetuning.csv")

# Save the fine-tuned model
torch.save(model.state_dict(), 'final_model3.pth')


# In[28]:


model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Load the trained model's parameters (state_dict)
model.load_state_dict(torch.load('final_model3.pth'))


model.to(device)


model.eval()

# Preprocess the input text
random_text = clean_tweet("Congratulations on your promotion! You truly deserve it.")

# Tokenize the input
inputs = tokenizer(
    random_text,
    add_special_tokens=True,
    max_length=80,  
    padding='max_length',
    return_attention_mask=True,
    return_tensors='pt',
    truncation=True
)


inputs = {key: value.to(device) for key, value in inputs.items()}


with torch.no_grad():
    outputs = model(**inputs)


logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

# Map the predicted class to sentiment label
labels = ['negative', 'neutral', 'positive']  
predicted_sentiment = labels[predicted_class]

# Output results
print(f"Text: {random_text}")
print(f"Predicted Sentiment: {predicted_sentiment}")





