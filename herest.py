import subprocess

def run_herest(
    config_file="config",
    phones_mlf="phones0.mlf",
    train_scp="train.scp",
    macros_file="hmm0/macros",
    hmmdefs_file="hmm0/hmmdefs",
    output_dir="hmm1",
    monophones_file="monophones0",
    pruning_threshold="250.0 150.0 1000.0",
    trace_level="1"
):
    command = [
        "HERest",
        "-A",
        "-D",
        "-T", trace_level,
        "-C", config_file,
        "-I", phones_mlf,
        "-t"] + pruning_threshold.split() + [
        "-S", train_scp,
        "-H", macros_file,
        "-H", hmmdefs_file,
        "-M", output_dir,
        monophones_file
    ]

    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        print("HERest command executed successfully.")
        print("Output:")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("Error executing HERest command:")
        print(e.stderr)
        if "Unknown label" in e.stderr:
            print("\nPossible solutions:")
            print("1. Check if the unknown label is in your phone list file.")
            print("2. Verify that the label is correctly used in your MLF file.")
            print("3. Ensure the label is properly defined in your HMM definitions.")
            print("4. Review your dictionary for any misuse of this label.")
    except FileNotFoundError:
        print("Error: HERest command not found. Make sure it's installed and in your system PATH.")

if __name__ == "__main__":
    run_herest(
        config_file="hcompv_config",
        phones_mlf="Datasets/train/phones0.mlf",
        train_scp="hcompv_train.scp",
        macros_file="hcompv_hmm0/macros",
        hmmdefs_file="hcompv_hmm0/hmmdefs",
        output_dir="hmm1",
        monophones_file="Datasets/train/monophones0",
        pruning_threshold="250.0 150.0 1000.0",
        trace_level="1"
    )