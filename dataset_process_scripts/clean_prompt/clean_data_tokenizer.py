import pandas as pd
import sys
import os
import math
from tqdm import tqdm

sys.path.insert(0, os.path.abspath('.'))
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

def check_prompts():
    file_path = "demo/metadata_vace.csv"
    print(f"Loading {file_path}")
    df = pd.read_csv(file_path)
    
    print("Loading HuggingfaceTokenizer...")
    try:
        tokenizer_path = "models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl/"
        tokenizer = HuggingfaceTokenizer(name=tokenizer_path, clean="canonicalize")
    except Exception as e:
        print(f"Failed to load tokenizer: {e}")
        return
        
    invalid_indices = []
    
    print("Processing prompts through tokenizer...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        prompt = row['prompt']
        
        # explicit float check that user reported
        if isinstance(prompt, float) or type(prompt) == float:
            invalid_indices.append(idx)
            print(f"Row {idx} is float: {prompt}")
            continue

        try:
            tokenizer(prompt)
        except Exception as e:
            invalid_indices.append(idx)
            print(f"Row {idx} failed with error: {e}")
                
    if len(invalid_indices) > 0:
        print(f"Found {len(invalid_indices)} erroneous rows. Removing them...")
        df_clean = df.drop(index=invalid_indices)
        df_clean.to_csv(file_path, index=False)
        print("Done. Cleaned dataset saved.")
    else:
        print("No errors found when processing strings.")

if __name__ == "__main__":
    check_prompts()
