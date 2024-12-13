import subprocess
import os

def run_hdman(wlist_file, monophones_file, dict_file, lexicon_file, log_file):
    # Ensure the required files exist
    required_files = [wlist_file, lexicon_file]
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Required file not found: {file}")

    # Construct the HDMan command
    hdman_command = [
        "HDMan",
        "-A",
        "-D",
        "-T","1",
        "-m",                  # Strip triphone contexts
        "-w", wlist_file,      # Word list file
        "-n", monophones_file, # Output monophones file
        "-l", log_file,        # Log file
        dict_file,             # Output dictionary file
        lexicon_file           # Input lexicon file
    ]

    try:
        # Execute HDMan
        result = subprocess.run(hdman_command, check=True, capture_output=True, text=True)
        
        # Print the output
        print("HDMan executed successfully.")
        print("Standard output:", result.stdout)
        
        # Check if the output files were created
        if os.path.isfile(dict_file) and os.path.isfile(monophones_file):
            print(f"Dictionary file created: {dict_file}")
            print(f"Monophones file created: {monophones_file}")
        else:
            print("Warning: Expected output files were not created.")
        
    except subprocess.CalledProcessError as e:
        print("Error executing HDMan:")
        print("Return code:", e.returncode)
        print("Standard error:", e.stderr)

# Example usage
wlist_file = "Datasets/train/wlist"
monophones_file = "Datasets/train/monophones1"
dict_file = "Datasets/train/dict"
lexicon_file = "VoxForgeDict.txt"
log_file = "Datasets/train/dlog"

run_hdman(wlist_file, monophones_file, dict_file, lexicon_file, log_file)