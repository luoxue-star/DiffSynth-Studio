import pandas as pd
import sys
import os
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

def process_and_clean():
    file_path = "demo/metadata_vace.csv"
    print(f"Loading metadata from {file_path}...")
    
    # Load with default parameters, exactly like UnifiedDataset reads it
    df = pd.read_csv(file_path)
    
    print("Loading text encoder HuggingfaceTokenizer...")
    # Use the local model path or HuggingFace repo path
    tokenizer_path = "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/"
    if not os.path.exists(tokenizer_path):
        tokenizer_path = "google/umt5-xxl"
        
    try:
        tokenizer = HuggingfaceTokenizer(name=tokenizer_path, clean="canonicalize")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        print("Please ensure you have internet access or the model is downloaded locally.")
        return
        
    invalid_indices = []
    
    print(f"Quickly processing all {len(df)} prompts through the text encoder...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        
        # 1. Direct float/NaN check: pandas converts empty cells to float NaNs
        if pd.isna(prompt) or isinstance(prompt, float):
            invalid_indices.append(idx)
            print(f"\n[Error] Row {idx} has invalid float/NaN prompt: {prompt}")
            continue

        # 2. Check for non-string prompts
        if not isinstance(prompt, str):
            invalid_indices.append(idx)
            print(f"\n[Error] Row {idx} has non-string prompt: {prompt} of type {type(prompt)}")
            continue

        # 3. Simulate exact processing through the text encoder
        try:
            tokenizer(prompt)
        except Exception as e:
            invalid_indices.append(idx)
            print(f"\n[Error] Row {idx} failed text encoding: {e}")
                
    print(f"\nIdentification complete. Found {len(invalid_indices)} erroneous rows.")
    
    if len(invalid_indices) > 0:
        print("Removing the identified erroneous data...")
        df_clean = df.drop(index=invalid_indices)
        df_clean.to_csv(file_path, index=False)
        print(f"Done. Successfully saved cleaned data to {file_path}.")
    else:
        print("No erroneous data found in the current CSV file.")

if __name__ == "__main__":
    process_and_clean()
