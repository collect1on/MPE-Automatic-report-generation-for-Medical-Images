# This script loads a 4-bit quantized Llama-3 model (via Unsloth) and a tokenizer,
# uses them to generate radiology report text based on partial findings,
# and then compares the generated reports with the original reports 
# from the Indiana chest X-ray dataset.


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
from unsloth import FastLanguageModel 

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Maximum sequence length for the model
max_seq_length = 2048

# List of available 4-bit quantized models from Unsloth
fourbit_models = [
    "unsloth/mistral-7b-v0.3-bnb-4bit",      
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/llama-3-8b-bnb-4bit",           
    "unsloth/llama-3-8b-Instruct-bnb-4bit",
    "unsloth/llama-3-70b-bnb-4bit",
    "unsloth/Phi-3-mini-4k-instruct",        
    "unsloth/Phi-3-medium-4k-instruct",
    "unsloth/mistral-7b-bnb-4bit",
    "unsloth/gemma-7b-bnb-4bit",             
]

# Load model and tokenizer (using Llama-3 8B 4-bit)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = torch.float16,
    load_in_4bit = True,
)

# Put model in inference mode
FastLanguageModel.for_inference(model)


# =======================
# Load and preprocess data
# =======================

# Base directory for images (not used yet in this script)
img_base_dir = '/group/ems010/sliu/images/images_normalized'

# Indiana chest X-ray reports and projections
reports_df = pd.read_csv('/group/ems010/sliu/indiana_reports.csv')
projections_df = pd.read_csv('/group/ems010/sliu/indiana_projections.csv')

# Custom dataset wrapper for image-report pairs
class ImageCaptionData:
    def __init__(self, reports_df, projections_df):
        self.reports_df = reports_df.set_index('uid')
        self.projections_df = projections_df.set_index('uid')
        # Only keep UIDs with non-empty findings
        self.uids = reports_df[reports_df.findings.notnull()].uid.unique()

    def get_sample(self):
        uid = np.random.choice(self.uids)
        images = list(self.projections_df.loc[[uid]]['filename'])
        projections = list(self.projections_df.loc[[uid]]['projection'])
        findings = self.reports_df.loc[uid]['findings']
        return uid, images, projections, findings

paired_dataset = ImageCaptionData(reports_df, projections_df)

# Split dataset into train/test UIDs
uids = projections_df.uid.unique()
train_uids, test_uids = train_test_split(uids, test_size=0.1, random_state=42)

# Tokenize findings (for training dataset)
findings_corpus = [tokenizer.encode(line) for line in reports_df[reports_df.uid.isin(train_uids)].findings.dropna()]

# PyTorch Dataset for Llama-3
class llama3Dataset(Dataset):
    def __init__(self, txt_list, max_length):
        # Truncate and append EOS token
        self.input_ids = [torch.tensor(t[:max_length-1] + [tokenizer.eos_token_id]) for t in txt_list]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

# Collate function to pad sequences
def collate_batch(batch):
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# Create dataset and dataloader
dataset = llama3Dataset(findings_corpus, max_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)


# =======================
# Inference Demo
# =======================

# Select some sample UIDs for inference
sample_reports = list(reports_df[reports_df.uid.isin([415, 538, 907, 980, 1048])].findings.dropna())

model.eval()
generated_texts = []

with torch.no_grad():
    for smp_report in sample_reports:
        # Use first 5 words of the report as prompt
        prompt_text = " ".join(smp_report.split()[:5])
        encoded_input = tokenizer.encode(prompt_text, return_tensors='pt')
        encoded_input = encoded_input.to(device) 

        # Generate continuation
        generated_sequences = model.generate(
            input_ids=encoded_input,
            max_length=100,   # Max total length of generated text
            num_return_sequences=1,
            temperature=0.7,  # Sampling temperature
            top_k=50,         # Top-k sampling
            top_p=0.95,       # Nucleus sampling
            no_repeat_ngram_size=2,  # Avoid repeated n-grams
            pad_token_id=tokenizer.eos_token_id
        )

        # Decode generated text
        generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

# Print comparison between original and generated reports
for smp, gen in zip(sample_reports, generated_texts):
    print("Sample report:\n", smp)
    print("Generated report:\n", gen)
    print("------------")
