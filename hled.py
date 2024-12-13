import subprocess
import os

def run_hled(input_mlf, output_mlf, edit_script, dictionary):
    # Construct the HLEd command
    hled_command = [
        "HLEd",
        "-A",  # Print command line arguments
        "-D",  # Display configuration variables
        "-T", "1",  # Set trace level to 1
        "-l", "'*'",  # Label file wildcard pattern
        "-d", dictionary,  # Specify the dictionary file
        "-i", output_mlf,  # Output MLF file name
        edit_script,  # Edit script file
        input_mlf  # Input word-level MLF file
    ]

    # Convert the command list to a string
    command_str = " ".join(hled_command)

    try:
        # Execute the HLEd command
        result = subprocess.run(command_str, shell=True, check=True, capture_output=True, text=True)
        
        # Print the command output
        print("HLEd command output:")
        print(result.stdout)
        
        # Check if the output file was created
        if os.path.exists(output_mlf):
            print(f"Successfully created {output_mlf}")
        else:
            print(f"Error: {output_mlf} was not created")
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing HLEd command: {e}")
        print("Error output:")
        print(e.stderr)

# Example usage
input_mlf = "Datasets/train/words.mlf"
output_mlf = "Datasets/train/phones1.mlf"
edit_script = "mkphones1.led"
dictionary = "Datasets/train/dict"

run_hled(input_mlf, output_mlf, edit_script, dictionary)

#RUN THE COMMAND LINE VERSION!