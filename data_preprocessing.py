import pandas as pd
import soundfile as sf
import numpy as np
import io
import pyarrow.parquet as pq

def convert_flac_to_wav(df, output_dir='Datasets/train/audio'):
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    wav_files = []  # to store file names of saved WAV files

    for index, row in df.iterrows():
        flac_data = row['audio']  # assuming this column contains raw audio data
        wav_file_name = f'{output_dir}/sample{index+1}.wav'

        # Convert FLAC raw data to WAV
        with io.BytesIO(flac_data['bytes']) as flac_buffer:
            data, samplerate = sf.read(flac_buffer)
            sf.write(wav_file_name, data, samplerate)
            wav_files.append(wav_file_name)

    return wav_files



# Sample usage:
# Assuming you have already loaded your LibriSpeech data into a DataFrame called 'librispeech_df'
# with a column 'audio_data' containing the raw FLAC audio data

# wav_files = convert_flac_to_wav(librispeech_df)
# print(f"Converted {len(wav_files)} files to WAV format.")

parquet_file = pq.ParquetFile("../Data/train_clean_100.parquet")

for i in parquet_file.iter_batches(batch_size=100):
        df = i.to_pandas()
        wav_files = convert_flac_to_wav(df)