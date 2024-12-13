import os
import re

def rename_samples(directory):
    # Compile a regular expression to match the file names
    pattern = re.compile(r'sample(\d+)\..*')
    
    # Get all files in the directory
    files = os.listdir(directory)
    
    # Sort the files to ensure we process them in order
    files.sort()
    
    for file in files:
        match = pattern.match(file)
        if match:
            # Extract the number from the filename
            old_num = int(match.group(1))
            
            # Calculate the new number (add 1)
            new_num = old_num + 1
            
            # Get the file extension
            _, extension = os.path.splitext(file)
            
            # Create the new filename
            new_name = f'sample{new_num:01d}{extension}'
            
            # Construct full file paths
            old_path = os.path.join(directory, file)
            new_path = os.path.join(directory, new_name)
            
            # Rename the file
            os.rename(old_path, new_path)
            print(f'Renamed: {file} -> {new_name}')

# Specify the directory containing the files
directory = 'Datasets/dev/audio'

# Call the function to rename the files
rename_samples(directory)