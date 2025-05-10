import os
import torch
import torchaudio
import soundfile as sf
import traceback
import random
import numpy as np
from tqdm import tqdm
from audioseal import AudioSeal
from attack import all_attacks, attack_methods, attack, restore_audio
import argparse

# Load models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
detector = AudioSeal.load_detector("audioseal_detector_16bits").to(device)
generator = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
    
def is_sample_completed(output_dir, base_name):
    """
    base_name: relative path like '2035/152373/2035-152373-0005'
    """
    output_path = os.path.join(output_dir, os.path.dirname(base_name))
    base_filename = os.path.basename(base_name)

    required_files = [
        f"{base_filename}_original.wav",
        f"{base_filename}_watermarked.wav",
        f"{base_filename}_message.txt",
    ] + [
        f"{base_filename}_attacked_{atk[0].replace('-', '')}.wav"
        for atk in all_attacks()
    ] + [
        f"{base_filename}_detect_{atk[0].replace('-', '')}.txt"
        for atk in all_attacks()
    ]

    for fname in required_files:
        if not os.path.exists(os.path.join(output_path, fname)):
            return False
    return True

    
@torch.no_grad()
def main(filelist_path, data_path, output_dir):
    # Read file list
    with open(filelist_path, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]
    # Process each audio file
    for file_path in tqdm(file_paths, desc="Processing files"):
        # file name without suffix
        filename = os.path.splitext(os.path.basename(file_path))[0]
        # "~/autodl-tmp/AudioSeal/VoxpopuliOutputResult/20110117-0900-PLENARY-10_en_seg320"
        file_output_dir = os.path.join(output_dir, filename)
        # rel_path = os.path.relpath(file_path, os.path.expanduser("~/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean"))
        rel_path = os.path.relpath(file_path, data_path)
        base_name = os.path.splitext(rel_path)[0]
        if is_sample_completed(output_dir, base_name):
            # print(f"Skipping {filename}, already completed.")
            continue
        
        try:
            # Load audio
            waveform, sr = torchaudio.load(file_path)
            if sr != 16000:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
                waveform = resampler(waveform)
                sr = 16000

            # Make mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            waveform = waveform.to(device)
            # Ensure 3D input (batch, channel, samples)
            if waveform.dim() == 2:
                waveform = waveform.unsqueeze(1)

            # Generate random watermark message
            msg = torch.randint(0, 2, (waveform.shape[0], generator.msg_processor.nbits), device=waveform.device)

            # Generate watermark
            watermark = generator.get_watermark(waveform, sr, message=msg)

            # Create watermarked audio
            watermarked_audio = waveform + watermark

            # Prepare output paths
            # rel_path = os.path.relpath(file_path, os.path.expanduser("~/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean"))
            base_name = os.path.splitext(rel_path)[0]
            output_subdir = os.path.join(output_dir, os.path.dirname(base_name))
            os.makedirs(output_subdir, exist_ok=True)

            # Save clean and watermarked audio
            # Save clean audio
            sf.write(
                os.path.join(output_subdir, f"{os.path.basename(base_name)}_original.wav"),
                waveform.squeeze().cpu().detach().numpy(), sr
            )

            # Save watermarked audio
            sf.write(
                os.path.join(output_subdir, f"{os.path.basename(base_name)}_watermarked.wav"),
                watermarked_audio.squeeze().cpu().detach().numpy(), sr
            )
            # Save the original watermark
            with open(os.path.join(output_subdir, f"{os.path.basename(base_name)}_message.txt"), "w") as f_out:
                f_out.write(f"Original Message: {msg.cpu().numpy().tolist()}\n")
                
            del waveform, watermark
            torch.cuda.empty_cache()

            # Run all attacks
            for attack_name, _ in all_attacks():
                attacked_audio, _ = attack(watermarked_audio, sr, [(attack_name, 1.0)])
                
                attacked_audio = restore_audio(attacked_audio, watermarked_audio.size(-1))
                # torch.backends.cudnn.enabled = False

                # Decode watermark from attacked audio
                result, recovered_msg = detector.detect_watermark(attacked_audio, sr)

                # Compute BER
                ber = (msg != recovered_msg).float().mean().item() * 100

                # Save attacked audio and detection result
                atk_tag = attack_name.replace("-", "")
                # sf.write(os.path.join(output_subdir, f"{os.path.basename(base_name)}_attacked_{atk_tag}.wav"), attacked_audio.squeeze().cpu().detach().numpy(), sr)
                with open(os.path.join(output_subdir, f"{os.path.basename(base_name)}_detect_{atk_tag}.txt"), "w") as f:
                    f.write(f"Recovered Message: {recovered_msg.cpu().numpy().tolist()}\n")
                    f.write(f"Bit Error Rate: {ber:.2f}%\n")
                    
                del result, recovered_msg
                torch.cuda.empty_cache()
                    
            del watermarked_audio, msg
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            traceback.print_exc()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filelist_path") 
    parser.add_argument("-d", "--data_path") 
    parser.add_argument("-o", "--output_dir")
    args = parser.parse_args()
    # Paths
    filelist_path = os.path.expanduser(args.filelist_path)
    data_path = os.path.expanduser(args.data_path)
    output_dir = os.path.expanduser(args.output_dir)
    os.makedirs(output_dir, exist_ok = True)

    main(filelist_path, data_path, output_dir)