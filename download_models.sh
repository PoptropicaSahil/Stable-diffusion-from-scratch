#!/bin/bash

# Create data directory if it doesn't exist
mkdir -p data

# URLs 
TOKENIZER_MERGES_URL="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_merges.txt"
TOKENIZER_VOCAB_URL="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/tokenizer/tokenizer_vocab.json"
# CHECKPOINT_URL="https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt"

# Download 
echo "Downloading tokenizer_merges.txt..."
wget -q -O data/tokenizer_merges.txt "$TOKENIZER_MERGES_URL"

echo "Downloading tokenizer_vocab.json..."
wget -q -O data/tokenizer_vocab.json "$TOKENIZER_VOCAB_URL"

echo "Downloading v1-5-pruned-emaonly.ckpt..."
# wget -q -O data/v1-5-pruned-emaonly.ckpt "$CHECKPOINT_URL"

echo "Download complete. Files are saved in the 'data' folder."