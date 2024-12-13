import os

def generate_train_scp(wav_dir, mfc_dir, output_file='dev_audio.scp'):
    """
    Generate a train.scp file with relative paths of wav and mfc files.
    
    Args:
    wav_dir (str): Directory containing .wav files
    mfc_dir (str): Directory to store .mfc files
    output_file (str): Name of the output .scp file
    """
    # Ensure the directories are absolute paths
    wav_dir = os.path.abspath(wav_dir)
    mfc_dir = os.path.abspath(mfc_dir)
    
    # Find all .wav files in the wav directory
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]
    
    # Sort the files to ensure consistent ordering
    wav_files.sort()
    
    # Open the output file for writing
    with open(output_file, 'w') as scp_file:
        for wav_file in wav_files:
            # Create corresponding .mfc filename
            mfc_file = os.path.splitext(wav_file)[0] + '.mfc'
            
            # Create relative paths
            wav_relative = os.path.relpath(os.path.join(wav_dir, wav_file))
            mfc_relative = os.path.relpath(os.path.join(mfc_dir, mfc_file))
            
            # Write to .scp file
            scp_file.write(f"{wav_relative} {mfc_relative}\n")
    
    print(f"train.scp file generated successfully at {output_file}")

# Example usage
wav_directory = 'Datasets/dev/audio'
mfc_directory = 'Datasets/dev/audio'
generate_train_scp(wav_directory, mfc_directory)