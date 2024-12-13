import os
import glob

def generate_file_lists(base_dir, dataset_type):
    audio_dir = os.path.join(base_dir, dataset_type, 'audio')
    transcript_dir = os.path.join(base_dir, dataset_type, 'transcripts')
    
    audio_files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
    
    audio_list_file = f'{dataset_type}_audio.scp'
    transcript_list_file = f'{dataset_type}_transcripts.scp'
    
    with open(audio_list_file, 'a') as audio_list, open(transcript_list_file, 'a') as transcript_list:
        for audio_file in audio_files:
            relative_path = os.path.relpath(audio_file, audio_dir)
            base_name = os.path.splitext(relative_path)[0]
            transcript_file = os.path.join(transcript_dir, base_name + '.lab')
            mfc_file = os.path.join(audio_dir, base_name + '.mfc')

            if os.path.exists(transcript_file):
                audio_list.write(f'{audio_file} {mfc_file}\n')
                transcript_list.write(f'{transcript_file}\n')
            else:
                print(f"Warning: No transcript found for {audio_file}")

def main():
    base_dir = 'Datasets'  # Replace with your actual base directory
    
    for dataset_type in ['train', 'dev', 'test']:
        print(f"Generating lists for {dataset_type} set...")
        generate_file_lists(base_dir, dataset_type)
    
    print("File lists generation complete.")

if __name__ == '__main__':
    main()