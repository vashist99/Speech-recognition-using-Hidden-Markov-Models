import pandas as pd
import os
import pyarrow.parquet as pq

def generate_htk_label_files(df, output_dir):
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each row in the DataFrame
    for index, row in df.iterrows():
        # Get the raw text for the current utterance
        utterance = row['transcript']
        
        # Clean and prepare the utterance for HTK format
        cleaned_utterance = clean_utterance(utterance)
        
        # Generate a unique filename for the .lab file
        lab_filename = f'sample{index+1}.lab'
        lab_filepath = os.path.join(output_dir, lab_filename)
        
        # Write the cleaned utterance to the .lab file
        with open(lab_filepath, 'w') as lab_file:
            lab_file.write(cleaned_utterance + '\n')
        
    print(f"Generated {len(df)} HTK-compatible label files in {output_dir}")

def clean_utterance(utterance):
    # Convert to uppercase (HTK typically uses uppercase)
    utterance = utterance.upper()
    
    # Remove punctuation and special characters
    utterance = ''.join(char for char in utterance if char.isalnum() or char.isspace())
    
    # Replace multiple spaces with a single space
    utterance = ' '.join(utterance.split())
    
    return utterance

# Example usage:
# Assuming you have your LibriSpeech data loaded into a DataFrame 'librispeech_df'
# with a column 'raw_text' containing the utterances

# Specify the output directory for the label files
output_directory = 'htk_labels'


parquet_file = pq.ParquetFile("../Data/train_clean_100.parquet")

for i in parquet_file.iter_batches(batch_size=100):
        df = i.to_pandas()
        generate_htk_label_files(df,'Datasets/train/transcripts')    
# Generate HTK-compatible label files
# generate_htk_label_files(librispeech_df, output_directory)