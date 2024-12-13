import os
import glob

# Specify the folder containing .lab files
folder_path = 'Datasets/test/transcripts'

# Get a list of all .lab files in the folder, sorted numerically
lab_files = sorted(glob.glob(os.path.join(folder_path, 'sample*.lab')), 
                   key=lambda x: int(x.split('.')[0][-1]))

# Open the output file
with open('Datasets/test/prompt.txt', 'w') as outfile:
    # Iterate through the sorted .lab files
    for i, lab_file in enumerate(lab_files):
        with open(lab_file, 'r') as infile:
            # Read the content of the .lab file
            content = infile.read().strip()
            # Write the formatted line to the output file
            outfile.write(f"*/sample{i+1} {content}\n")

print("Processing complete. Check 'prompt.txt' for the output.")