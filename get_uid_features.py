# This scripts loads text and author labels from a .csv/any data source,
# calculates all UID features needed for GPT-who
# and writes it to a new .csv 
# which is the input to "gpt-who.py"

from collections import Counter
import transformers
import pandas as pd
import numpy as np
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import gc
from collections import Counter
import re
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str)
parser.add_argument('--cache_path', type=str, default="./.cache/models/gpt2-xl")
parser.add_argument('--output_path', type=str, default = "./scores/uid_features.csv" )

args = parser.parse_args()

# For custom datasets, any "input_file" i.e. csv with text and corresponding label
# can be loaded into a DataFrame "df" with two names columns:
# 1. "text" and 2. "label"
input_file = args.input_path 

# Load data to DataFrame
df = pd.read_csv(input_file)

device = torch.device('cuda')
print("CUDA status:", torch.cuda.is_available())

all_sents = df['text']
all_labels = df['label']

# Functions to get UID_Diff & UID_Diff2 features
def local_diff(x):
    d = 0
    for i in range(len(x)-1):
        d += abs(x[i+1]-x[i])
    return d/len(x)

def local_diff2(x):
    d = 0
    for i in range(len(x)-1):
        d += (x[i+1]-x[i])**2
    return d/len(x)

    
tokenizer_class, model_class = GPT2Tokenizer, GPT2LMHeadModel
tokenizer = tokenizer_class.from_pretrained("gpt2-xl", cache_dir=args.cache_path)
model = model_class.from_pretrained("gpt2-xl", cache_dir=model_checkpoint).to(device)
tokenizer.pad_token = tokenizer.eos_token

# Getting UID_var and all surprisals for "UID spans" features
def get_line_uid_surp(lines):
    with torch.no_grad():
        lines = tokenizer.eos_token + lines
        tok_res = tokenizer(lines, return_tensors='pt')
        input_ids = tok_res['input_ids'][0].to(device)
        attention_mask = tok_res['attention_mask'][0].to(device)
        lines_len = torch.sum(tok_res['attention_mask'], dim=1)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
        loss, logits = outputs[:2]
        #print("lines len", lines_len)
    line_log_prob = torch.tensor([0.0])
    word_probs = []
    for token_ind in range(lines_len - 1):
        token_prob = F.softmax(logits[token_ind], dim=0).to('cpu')
        token_id = input_ids[token_ind + 1].to('cpu')
        target_log_prob = -torch.log(token_prob[token_id]).detach().cpu().numpy()
        line_log_prob += -torch.log(token_prob[token_id]).detach().cpu().numpy()
        word_probs.append(target_log_prob)
    
    # mu = average surprisal 
    mu = line_log_prob/(lines_len-1)
    uid = torch.tensor([0.0])

    for i in range(len(word_probs)):
        uid += (word_probs[i] - mu)**2/(len(word_probs))
    sentence_uids = uid.detach().cpu().numpy()[0]
    sentence_surprisal = np.mean(word_probs)
    sentence_length = lines_len.detach().cpu().numpy()[0]
    torch.cuda.empty_cache()
    return(sentence_uids, sentence_surprisal, word_probs, sentence_length)

output_file = args.output_path
with open(output_file, "w") as f:
    writer = csv.writer(f)
    row=["text", "label", "uid_var", "uid_diff", "uid_diff2", "mean", "sum", "surps", "n_token"]
    writer.writerow(row)

for i in tqdm(range(len(all_sents))):
    batch = all_sents[i] 
    labels = all_labels[i]
    uids, surps, probs, lens = get_line_uid_surp(batch)
    uid_diff1 = local_diff(probs)
    uid_diff2 = local_diff2(probs)
    sum = np.sum(probs)
    gc.collect()
    torch.cuda.empty_cache()
    row_=[batch, labels, uids, uid_diff1, uid_diff2, surps, sum, probs, lens]

    with open(output_file, "a") as f:
        writer = csv.writer(f)
        writer.writerow(row_)

            