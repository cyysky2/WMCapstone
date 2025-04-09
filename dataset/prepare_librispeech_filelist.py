import os
import glob
import subprocess
import argparse

def convert_flac_to_wav_24k(flac_path, wav_path):
    os.makedirs(os.path.dirname(wav_path), exist_ok=True)
    command = ['ffmpeg', '-y', '-i', flac_path, '-ar', '24000', wav_path]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_dataset(input_dir, output_dir, file_list_path):
    flac_files = glob.glob(os.path.join(input_dir, '**/*.flac'), recursive=True)
    with open(file_list_path, 'w', encoding='utf-8') as f:
        for flac_path in flac_files:
            relative_path = os.path.relpath(flac_path, input_dir)
            wav_path = os.path.join(output_dir, relative_path).replace('.flac', '.wav')
            convert_flac_to_wav_24k(flac_path, wav_path)
            f.write(os.path.abspath(wav_path) + '\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='123')
    a = parser.parse_args()

    # Paths to original LibriSpeech data
    train_input_dir = 'LibriSpeech/LibriSpeech/train-clean-360'
    val_input_dir = 'LibriSpeech/LibriSpeech/dev-clean'
    test_input_dir = 'LibriSpeech/LibriSpeech/test-clean'

    # Output directories for wavs
    train_output_dir = 'LibriSpeech/LibriSpeech_24k_wav/train-clean-360'
    val_output_dir = 'LibriSpeech/LibriSpeech_24k_wav/dev-clean'
    test_output_dir = 'LibriSpeech/LibriSpeech_24k_wav/test-clean'

    # Output file list paths
    train_list_path = 'train_filelist.txt'
    val_list_path = 'val_filelist.txt'
    test_list_path = 'test_filelist.txt'

    if '1' in a.stage:
        print("Processing training set...")
        process_dataset(train_input_dir, train_output_dir, train_list_path)
    if '2' in a.stage:
        print("Processing validation set...")
        process_dataset(val_input_dir, val_output_dir, val_list_path)
    if '3' in a.stage:
        print("Processing test set...")
        process_dataset(test_input_dir, test_output_dir, test_list_path)

    print("Done! File lists created:")
    print(f"Training list: {train_list_path}")
    print(f"Validation list: {val_list_path}")
    print(f"Testing list: {test_list_path}")
