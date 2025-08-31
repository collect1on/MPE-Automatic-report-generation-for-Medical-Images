import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.translate.bleu_score import sentence_bleu
from transformers import AutoModelForCausalLM, AutoTokenizer
import csv

# -------------------------------
# Set CUDA device (optional)
# -------------------------------
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.cuda.set_device(1)  # alternative method

# -------------------------------
# Define device (GPU if available, else CPU)
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# -------------------------------
# Load tokenizer and fine-tuned model
# -------------------------------
# Replace paths with descriptive placeholders for sharing
tokenizer = AutoTokenizer.from_pretrained("path/to/fine-tuned-llama3-tokenizer")
model = AutoModelForCausalLM.from_pretrained("path/to/fine-tuned-llama3-model")
# model = model.to(device)  # Move model to GPU if needed

# -------------------------------
# Load dataset
# -------------------------------
# Replace with actual dataset paths
reports_df = pd.read_csv("path/to/indiana_reports.csv")
projections_df = pd.read_csv("path/to/indiana_projections.csv")

# -------------------------------
# Select subset of reports for evaluation
# -------------------------------
train_uids = sorted(reports_df['uid'].unique())[:200]  # Take first 200 unique UIDs
sample_reports = list(reports_df[reports_df.uid.isin(train_uids)].findings.dropna())

# -------------------------------
# Define BLEU score calculation function
# -------------------------------
def calculate_bleu_scores(reference, candidate):
    """
    Calculate BLEU-1, BLEU-2, BLEU-3 scores between reference and generated text
    """
    bleu1 = sentence_bleu([reference.split()], candidate.split(), weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([reference.split()], candidate.split(), weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu([reference.split()], candidate.split(), weights=(0.33, 0.33, 0.33, 0))
    return bleu1, bleu2, bleu3

# -------------------------------
# Generate text and calculate BLEU scores
# -------------------------------
model.eval()
generated_texts = []
bleu_scores = []

with torch.no_grad():
    for smp_report in sample_reports:
        # Take first 5 words of the report as prompt
        prompt_text = " ".join(smp_report.split()[:5])
        encoded_input = tokenizer.encode(prompt_text, return_tensors='pt').to(device)
        
        # Generate text using the model
        generated_sequences = model.generate(
            input_ids=encoded_input,
            max_length=100,             # Total length including prompt
            num_return_sequences=1,     # Generate 1 sequence
            temperature=0.7,            # Sampling temperature
            top_k=50,                   # Top-k filtering
            top_p=0.95,                 # Top-p (nucleus) filtering
            no_repeat_ngram_size=2,     # Prevent repeated n-grams
            pad_token_id=tokenizer.eos_token_id
        )
        generated_text = tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
        generated_texts.append(generated_text)

        # Calculate BLEU scores
        bleu1, bleu2, bleu3 = calculate_bleu_scores(smp_report, generated_text)
        bleu_scores.append((bleu1, bleu2, bleu3))

# -------------------------------
# Compute average BLEU scores
# -------------------------------
bleu1_avg = np.mean([score[0] for score in bleu_scores])
bleu2_avg = np.mean([score[1] for score in bleu_scores])
bleu3_avg = np.mean([score[2] for score in bleu_scores])

# -------------------------------
# Prepare data for CSV output
# -------------------------------
csv_data = []
for i, (smp_report, gen_report, (bleu1, bleu2, bleu3)) in enumerate(zip(sample_reports, generated_texts, bleu_scores)):
    csv_data.append({
        'uid': train_uids[i],
        'sample_report': smp_report,
        'generated_report': gen_report,
        'BLEU-1': bleu1,
        'BLEU-2': bleu2,
        'BLEU-3': bleu3,
        'BLEU-1avg': bleu1_avg,
        'BLEU-2avg': bleu2_avg,
        'BLEU-3avg': bleu3_avg
    })

# -------------------------------
# Write results to CSV file
# -------------------------------
output_csv = 'generated_reports_with_bleu_llama3.csv'
with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.DictWriter(file, fieldnames=[
        'uid', 'sample_report', 'generated_report',
        'BLEU-1', 'BLEU-2', 'BLEU-3',
        'BLEU-1avg', 'BLEU-2avg', 'BLEU-3avg'
    ])
    writer.writeheader()
    writer.writerows(csv_data)

print(f"Results written to {output_csv}")
