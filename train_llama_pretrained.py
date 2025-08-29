import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tqdm
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load tokenizer and model from pretrained weights
model_path = "your_pretrained_model_path"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# Move model to GPU if available
model = model.to(device)

# Data preparation - replace with your actual data paths
img_base_dir = 'your_image_directory_path'
reports_df = pd.read_csv('your_reports_csv_path')
projections_df = pd.read_csv('your_projections_csv_path')

class ImageCaptionData:
    def __init__(self, reports_df, projections_df):
        self.reports_df = reports_df.set_index('uid')
        self.projections_df = projections_df.set_index('uid')
        self.uids = reports_df[reports_df.findings.notnull()].uid.unique()

    def get_sample(self):
        uid = np.random.choice(self.uids)
        images = list(self.projections_df.loc[[uid]]['filename'])
        projections = list(self.projections_df.loc[[uid]]['projection'])
        findings = self.reports_df.loc[uid]['findings']
        return uid, images, projections, findings

# Create dataset instance
paired_dataset = ImageCaptionData(reports_df, projections_df)
uids = projections_df.uid.unique()
train_uids, test_uids = train_test_split(uids, test_size=0.1, random_state=42)

# Prepare text corpus for training
findings_corpus = [tokenizer.encode(line) for line in reports_df[reports_df.uid.isin(train_uids)].findings.dropna()]

class llama3Dataset(Dataset):
    def __init__(self, txt_list, max_length):
        self.input_ids = [torch.tensor(t[:max_length-1] + [tokenizer.eos_token_id]) for t in txt_list]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

def collate_batch(batch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# Create dataset and dataloader
dataset = llama3Dataset(findings_corpus, max_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

# Training setup
model.train()
optimizer = AdamW(model.parameters(), lr=5e-5)

NUM_EPOCHS = 3

# Training loop
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in tqdm.auto.tqdm(dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        optimizer.zero_grad()
        
        # Move batch to device
        batch = batch.to(device)
        
        # Forward pass with teacher forcing
        outputs = model(batch, labels=batch)
        
        # Calculate loss and backpropagate
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    print(f'Epoch {epoch+1} completed - Average Loss: {avg_loss:.4f}')

# Save fine-tuned model and tokenizer
output_path = "your_output_directory_path"
model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("Model and tokenizer saved successfully.")
print(f"Model saved to: {output_path}")
