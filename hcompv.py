#!!!!!! Use the raw command
import subprocess
import os
# import logging

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_hcompv(config_file, scp_file, output_dir, proto_file):
    # os.makedirs(output_dir, exist_ok=True)

    hcompv_command = [
        "HCompV",
        "-A",
        "-D",
        "-T", "1",
        "-C", config_file,
        "-f", "0.01",
        "-m",
        "-S", scp_file,
        "-M", output_dir,
        proto_file
    ]

    command_str = " ".join(hcompv_command)

    try:
        result = subprocess.run(command_str, check=True, capture_output=True, text=True)
        # logging.info(f"HCompV executed successfully. Output saved in {output_dir}")
        print("HCompV command output:")
        print(result.stdout)

        if os.path.exists(output_dir):
            print(f"Successfully created {output_dir}")
        else:
            print(f"Error: {output_dir} was not created")
        # logging.debug(f"HCompV output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing HLEd command: {e}")
        print("Error output:")
        print(e.stderr)
        # logging.error(f"Error executing HCompV: {e}")
        # logging.error(f"HCompV stderr: {e.stderr}")
        # You might want to raise an exception here or handle the error in some way
    # except Exception as e:
    #     logging.error(f"Unexpected error: {e}")

config_file = "hcompv_config"
scp_file = "hcompv_train.scp"
output_dir = "hcompv_hmm0"
proto_file = "hcompv_proto"

run_hcompv(config_file, scp_file, output_dir, proto_file)