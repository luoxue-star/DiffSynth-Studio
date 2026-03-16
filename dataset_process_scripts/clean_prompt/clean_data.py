import pandas as pd
import sys
import os

# Add diffsynth to sys.path
sys.path.insert(0, os.path.abspath('.'))
from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

def process_and_clean():
    file_path = "demo/metadata_vace.csv"
    print(f"Loading {file_path}...")
    df = pd.read_csv(file_path)
    
    # We will simulate the same exact error handling happening in the module
    # To avoid downloading tokenizer, let's just use empty model if possible,
    # or just replicate the exact preprocessing logic:
    
    invalid_indices = []
    
    # Try passing each prompt
    for idx, row in df.iterrows():
        prompt = row['prompt']
        
        # simulated logic from line 313 in wan_video_text_encoder.py
        sequence = prompt
        try:
            if isinstance(sequence, str):
                sequence = [sequence]
            
            # self.clean check
            # sequence = [self._clean(u) for u in sequence]
            # Since we only want to catch the specific iterable error:
            sequence = [u for u in sequence]
        except TypeError as e:
            if "'float' object is not iterable" in str(e):
                print(f"Row {idx} caused TypeError: {e}. Data: {prompt}")
                invalid_indices.append(idx)
            else:
                raise e

    print(f"\nIdentified {len(invalid_indices)} erroneous rows.")
    
    if len(invalid_indices) > 0:
        print("Removing these rows...")
        df_clean = df.drop(index=invalid_indices)
        df_clean.to_csv(file_path, index=False)
        print("Cleaned CSV saved.")
    else:
        print("No errors found.")

if __name__ == "__main__":
    process_and_clean()
