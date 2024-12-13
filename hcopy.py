import subprocess
import sys

def run_hcopy(config_file, script_file):
    try:
        # Construct the HCopy command
        hcopy_command = ["HCopy", "-A","-D","-T","1","-C", config_file, "-S", script_file]
        
        # Run the HCopy command
        result = subprocess.run(hcopy_command, capture_output=True, text=True, check=True)
        
        # Print the output
        print("HCopy command executed successfully.")
        print("Output:")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"Error executing HCopy: {e}")
        print("Error output:")
        print(e.stderr)
    except FileNotFoundError:
        print("Error: HCopy command not found. Make sure HTK is installed and in your system PATH.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_hcopy.py <config_file> <script_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    script_file = sys.argv[2]
    
    run_hcopy(config_file, script_file)