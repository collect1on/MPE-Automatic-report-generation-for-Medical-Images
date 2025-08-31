# This program loads a fine-tuned Llama model and its corresponding tokenizer,
# uses the first few words from complete reports in the Indiana chest X-ray dataset as prompts,
# generates subsequent radiology findings text, and compares them with the original complete reports.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------
# Device setup
# ---------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------
# Load tokenizer and model (replace with your paths)
# ---------------------------------------------
tokenizer = AutoTokenizer.from_pretrained("PATH_TO_FINETUNED_LLAMA_TOKENIZER")
model = AutoModelForCausalLM.from_pretrained("PATH_TO_FINETUNED_LLAMA_MODEL")

# NOTE: Uncomment this if you want to move the model to GPU
# model = model.to(device)

# ---------------------------------------------
# Load datasets (Indiana chest X-ray dataset)
# ---------------------------------------------
img_base_dir = "PATH_TO_CHEST_XRAY_IMAGES"  # Directory containing normalized chest X-ray images
reports_df = pd.read_csv("PATH_TO_INDIANA_REPORTS_CSV")  # CSV file with reports
projections_df = pd.read_csv("PATH_TO_INDIANA_PROJECTIONS_CSV")  # CSV file with projection info

# ---------------------------------------------
# Data wrapper for pairing images with reports
# ---------------------------------------------
class ImageCaptionData:
    def __init__(self, reports_df, projections_df):
        # Store reports and projections with 'uid' as index
        self.reports_df = reports_df.set_index('uid')
        self.projections_df = projections_df.set_index('uid')
        # Keep only reports that have findings
        self.uids = reports_df[reports_df.findings.notnull()].uid.unique()

    def get_sample(self):
        """Randomly pick one sample consisting of uid, images, projections, and findings."""
        uid = np.random.choice(self.uids)
        images = list(self.projections_df.loc[[uid]]['filename'])
        projections = list(self.projections_df.loc[[uid]]['projection'])
        findings = self.reports_df.loc[uid]['findings']
        return uid, images, projections, findings

# Create dataset wrapper
paired_dataset = ImageCaptionData(reports_df, projections_df)

# Train/test split based on unique study IDs
uids = projections_df.uid.unique()
train_uids, test_uids = train_test_split(uids, test_size=0.1, random_state=42)

# Tokenize findings for training corpus
findings_corpus = [tokenizer.encode(line) for line in reports_df[reports_df.uid.isin(train_uids)].findings.dropna()]

# ---------------------------------------------
# Custom PyTorch Dataset for Llama training
# ---------------------------------------------
class llama3Dataset(Dataset):
    def __init__(self, txt_list, max_length):
        # Truncate to max_length - 1, then append EOS token
        self.input_ids = [torch.tensor(t[:max_length-1] + [tokenizer.eos_token_id]) for t in txt_list]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

# Collate function for padding sequences in a batch
def collate_batch(batch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# Create dataset and dataloader
dataset = llama3Dataset(findings_corpus, max_length=128)  # You can adjust max_length
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

# ---------------------------------------------
# Choose sample reports for evaluation
# ---------------------------------------------
# Example: Select findings for specific UIDs
sample_reports = list(reports_df[reports_df.uid.isin([410, 538, 907, 980, 1048])].findings.dropna())

# ---------------------------------------------
# Generate text from the model
# ---------------------------------------------
model.eval()
generated_texts = []
with torch.no_grad():
    for smp_report in sample_reports:
        # Use the first 5 words of the report as the prompt
        prompt_text = " ".join(smp_report.split()[:5])
        encoded_input = tokenizer.encode(prompt_text, return_tensors='pt').to(device)

        # Generate continuation
        generated_sequences = model.generate(
            input_ids=encoded_input,
            max_length=100,              # Total length including prompt
            num_return_sequences=1,      # Only one output per input
            temperature=0.7,             # Controls randomness
            top_k=50,                    # Top-k sampling
            top_p=0.95,                  # Nucleus (top-p) sampling
            no_repeat_ngram_size=2,      # Avoid repeating n-grams
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode generated tokens into text
        generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

# ---------------------------------------------
# Compare original and generated reports
# ---------------------------------------------
for smp, gen in zip(sample_reports, generated_texts):
    print("Sample report:\n", smp)
    print("Generated report:\n", gen)
    print("------------")
