import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import tqdm
from torch.optim import AdamW
from unsloth import FastLanguageModel 
from unsloth import is_bfloat16_supported
import torch
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, ConvNextImageProcessor
from torchvision.io import read_image, ImageReadMode
from transformers.models.convnext.feature_extraction_convnext import ConvNextFeatureExtractor
max_seq_length = 2048  # Supports RoPE Scaling internally, so choose any!
from torchvision import datasets, models, transforms
from transformers import AutoImageProcessor, ConvNextV2Model
import torch
from datasets import load_dataset

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

# Set device
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device("cpu")
    print("Using CPU")

# Define paths
img_base_dir = 'your_image_directory_path'
reports_df = pd.read_csv('your_reports_csv_path')
projections_df = pd.read_csv('your_projections_csv_path')

# Create dataset instance
paired_dataset = ImageCaptionData(reports_df, projections_df)
uid, images, projections, findings = paired_dataset.get_sample()

# Split data into train and test
uids = projections_df.uid.unique()
train_uids, test_uids = train_test_split(uids, test_size=0.1, random_state=42)

# Load model and tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = torch.float32,
    load_in_4bit = True,
)

# Enable model for training
FastLanguageModel.for_training(model)

# Prepare findings corpus for training
findings_corpus = [tokenizer.encode(line) for line in reports_df[reports_df.uid.isin(train_uids)].findings.dropna()]

class llama3Dataset(Dataset):
    def __init__(self, txt_list, max_length):
        self.input_ids = [torch.tensor(t[:max_length-1] + [tokenizer.eos_token_id]) for t in txt_list]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return self.input_ids[index]

def collate_batch(batch):
    """
    Pad the batch to ensure all sequences have the same length.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    return pad_sequence(batch, batch_first=True, padding_value=pad_id)

# Create dataset
dataset = llama3Dataset(findings_corpus, max_length=128)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

# Load language model
lang_model = model

# Load vision model and feature extractor
feature_extractor = ConvNextImageProcessor.from_pretrained(
    "facebook/convnextv2-tiny-1k-224",
    token='your_huggingface_token'
)
vision_model = ConvNextV2Model.from_pretrained("facebook/convnextv2-tiny-1k-224")

# Define constants
MAX_SEQ_LENGTH = 128
MAX_IMG_STACK = 5 + 1
BATCH_SIZE = 4

class ImageCaptionData(Dataset):
    def __init__(self, reports_df, projections_df, max_length=MAX_SEQ_LENGTH, max_img_stack=MAX_IMG_STACK):
        self.reports_df = reports_df.dropna(subset={'findings'}).set_index('uid')
        self.projections_df = projections_df.set_index('uid')
        self.uids = list(set(self.reports_df.index).intersection(self.projections_df.index))
        self.max_length = max_length
        self.max_img_stack = max_img_stack
        
    def __getitem__(self, index):
        uid = self.uids[index]
        image_paths = [os.path.join(img_base_dir, x) for x in list(self.projections_df.loc[[uid]]['filename'])]
        
        # Process images
        batch_features = [feature_extractor(read_image(img, ImageReadMode.RGB)).convert_to_tensors(tensor_type='pt') for img in image_paths]
        lst = [bf['pixel_values'][0] for bf in batch_features]
        images = torch.stack(lst)
        
        # Pad image stack
        len_image_stack = len(image_paths)
        img_dim = images[0].shape
        images = torch.cat((torch.zeros((self.max_img_stack - len_image_stack, *img_dim)), images), dim=0)
        
        # Tokenize findings
        findings = tokenizer.encode(self.reports_df.loc[uid]['findings'])
        findings = torch.tensor(findings[:self.max_length-1] + [tokenizer.eos_token_id])
        return len_image_stack, images, findings
    
    def __len__(self):
        return len(self.uids)
    
def collate_batch(batch):
    len_images = torch.tensor([item[0] for item in batch])
    img_data = torch.stack([item[1] for item in batch])
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
        
    tokens = pad_sequence([item[2] for item in batch], batch_first=True, padding_value=pad_id)
    labels = pad_sequence([torch.cat((torch.tensor([-100]*MAX_IMG_STACK), item[2])) for item in batch], batch_first=True, padding_value=pad_id) 
    return [len_images, img_data, tokens, labels, pad_sequence([torch.ones(len(item[2])+MAX_IMG_STACK) for item in batch], batch_first=True, padding_value=0)]

# Create train and test datasets
train_dataset = ImageCaptionData(reports_df[reports_df.uid.isin(train_uids)], projections_df[projections_df.uid.isin(train_uids)])
test_dataset = ImageCaptionData(reports_df[~reports_df.uid.isin(train_uids)], projections_df[~projections_df.uid.isin(train_uids)])

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)

# Define model dimensions
VISION_MODEL_OUTPUT_DIM = 768
LANG_MODEL_INPUT_DIM = 4096

class ProjectionModel(nn.Module):
    def __init__(self, vision_out_dim, lang_inp_dim):
        super(ProjectionModel, self).__init__()
        self.lin = nn.Linear(vision_out_dim, lang_inp_dim, bias=True)
    
    def forward(self, x):
        x = nn.functional.tanh(self.lin(x))
        return x
    
projection_model = ProjectionModel(VISION_MODEL_OUTPUT_DIM, LANG_MODEL_INPUT_DIM)

# Move models to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vision_model = vision_model.to(device)
projection_model = projection_model.to(device)

# Freeze vision model
vision_model.eval()
for param in vision_model.parameters():
    param.requires_grad = False
    
# Set up optimizer
optimizer = torch.optim.Adam(
    [
        {"params": lang_model.parameters(), "lr": 2e-5},
        {"params": projection_model.parameters(), "lr": 5e-5}
    ]
)

# Training setup
NUM_EPOCHS = 5
MODEL_CHECKPOINTS_PATH = 'your_model_checkpoint_path'
if not os.path.exists(MODEL_CHECKPOINTS_PATH):
    os.makedirs(MODEL_CHECKPOINTS_PATH)

best_val_loss = np.inf
train_loss = []
val_loss = []

# Training loop
for i in range(NUM_EPOCHS):
    train_batch_loss = [] 
    
    lang_model.train()
    projection_model.train()
    for l_img, img, tokens, labels, attn in tqdm.auto.tqdm(train_dataloader):
        img, tokens, labels, attn = img.to(device), tokens.to(device), labels.to(device), attn.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            # Get BOS embedding
            bos_embedding = lang_model.get_input_embeddings()(torch.tensor([tokenizer.bos_token_id]*MAX_IMG_STACK).to(device))
            bos_embedding = torch.stack([bos_embedding]*len(l_img))
            mask = torch.stack([torch.cat([torch.ones(MAX_IMG_STACK-limg), torch.zeros(limg)]).repeat(LANG_MODEL_INPUT_DIM,1).T for limg in l_img]).to(device)
            
            # Extract image features
            img_embed = vision_model(img.flatten(0,1)).pooler_output
            
        # Project image features to language model space
        img_embed = projection_model(img_embed).reshape(len(l_img), MAX_IMG_STACK, LANG_MODEL_INPUT_DIM)
        img_embed = bos_embedding*mask + (1-mask)*img_embed  # Replace missing images with BOS embedding

        # Get token embeddings
        tok_embed = lang_model.get_input_embeddings()(tokens)
        input_embed = torch.cat((img_embed, tok_embed), dim=1)
        
        # Forward pass
        outputs = lang_model(
            inputs_embeds=input_embed,
            labels=labels,
            attention_mask=attn,
        )

        # Backward pass
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item()/len(l_img))
        
    train_loss.append(np.mean(train_batch_loss))
    
    # Validation
    lang_model.eval()
    projection_model.eval()
    aggregated_val_loss = []
    with torch.no_grad():
        for l_img, img, tokens, labels, attn in tqdm.auto.tqdm(test_dataloader):
            img, tokens, labels, attn = img.to(device), tokens.to(device), labels.to(device), attn.to(device)
            
            # Get BOS embedding
            bos_embedding = lang_model.get_input_embeddings()(torch.tensor([tokenizer.bos_token_id]*MAX_IMG_STACK).to(device))
            bos_embedding = torch.stack([bos_embedding]*len(l_img))
            mask = torch.stack([torch.cat([torch.ones(MAX_IMG_STACK-limg), torch.zeros(limg)]).repeat(LANG_MODEL_INPUT_DIM,1).T for limg in l_img]).to(device)
            
            # Extract image features
            img_embed = vision_model(img.flatten(0,1)).pooler_output
        
            # Project image features
            img_embed = projection_model(img_embed).reshape(len(l_img), MAX_IMG_STACK, LANG_MODEL_INPUT_DIM)
            img_embed = bos_embedding*mask + (1-mask)*img_embed

            # Get token embeddings
            tok_embed = lang_model.get_input_embeddings()(tokens)
            input_embed = torch.cat((img_embed, tok_embed), dim=1)
            
            # Forward pass
            outputs = lang_model(
                inputs_embeds=input_embed,
                labels=labels,
                attention_mask=attn,
            )

            loss = outputs.loss
            aggregated_val_loss.append(loss.item()/len(l_img))
    
    # Print epoch results
    print("Epoch:", i, "Train Loss: {:.4f}".format(np.mean(train_batch_loss)), "Val Loss: {:.4f}".format(np.mean(aggregated_val_loss)))

    # Save model checkpoints
    epoch_checkpoint_path = os.path.join(MODEL_CHECKPOINTS_PATH, 'epoch_'+str(i))
    if not os.path.exists(epoch_checkpoint_path):
        os.makedirs(epoch_checkpoint_path)
    model.save_pretrained(os.path.join(epoch_checkpoint_path, 'model'))
    tokenizer.save_pretrained(os.path.join(epoch_checkpoint_path, 'tokenizer'))
    torch.save(projection_model.state_dict(), os.path.join(epoch_checkpoint_path, 'projection_model.pth'))
    
    # Update best validation loss and save the best model
    if (np.mean(aggregated_val_loss) < best_val_loss):
        best_val_loss = np.mean(aggregated_val_loss)
        best_checkpoint_path = os.path.join(MODEL_CHECKPOINTS_PATH, 'best_model')
        if not os.path.exists(best_checkpoint_path):
            os.makedirs(best_checkpoint_path)
        model.save_pretrained(os.path.join(best_checkpoint_path, 'model'))
        tokenizer.save_pretrained(os.path.join(best_checkpoint_path, 'tokenizer'))
        torch.save(projection_model.state_dict(), os.path.join(best_checkpoint_path, 'projection_model.pth'))
        print("Saved best checkpoint at " + best_checkpoint_path)
        
    # Save loss results to a file
    np.savez(os.path.join(MODEL_CHECKPOINTS_PATH, 'losses.npz'), train_loss=train_loss, val_loss=val_loss)
    
    # Update learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.2
