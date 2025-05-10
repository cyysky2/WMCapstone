import os
import numpy as np
import torchaudio
import soundfile as sf
import torch
from tqdm import tqdm
import wavmark
from wavmark.utils import file_reader
import traceback
from attack import all_attacks, attack_methods, attack, restore_audio
import argparse

# Resume check
def is_sample_completed(output_dir, base_name):
    path = os.path.join(output_dir, os.path.dirname(base_name))
    fname = os.path.basename(base_name)
    required = [f"{fname}_original.wav", f"{fname}_watermarked.wav", f"{fname}_payload.txt"] + \
               [f"{fname}_attacked_{atk[0].replace('-', '')}.wav" for atk in all_attacks()] + \
               [f"{fname}_detect_{atk[0].replace('-', '')}.txt" for atk in all_attacks()]
    return all(os.path.exists(os.path.join(path, f)) for f in required)

# Main loop
@torch.no_grad()
def main(file_paths, data_path, output_dir):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = wavmark.load_model().to(device)

    with open(filelist_path, "r") as f:
        file_paths = [line.strip() for line in f if line.strip()]

    for file_path in tqdm(file_paths, desc="Processing"):
        try:
            rel_path = os.path.relpath(file_path, data_path)
            base_name = os.path.splitext(rel_path)[0]
            output_subdir = os.path.join(output_dir, os.path.dirname(base_name))
            os.makedirs(output_subdir, exist_ok=True)
            base_filename = os.path.basename(base_name)

            if is_sample_completed(output_dir, base_name):
                print(f"Skipping {base_filename} (already done)")
                continue

            # Step 1: Load and preprocess
            signal = file_reader.read_as_single_channel(file_path, aim_sr=16000)
            # Make sure the signal is exactly 16000 samples long
            target_len = 16000
            if len(signal) > target_len:
                signal = signal[:target_len]
            elif len(signal) < target_len:
                pad = target_len - len(signal)
                signal = np.pad(signal, (0, pad), mode='constant')
                
            payload = np.random.choice([0, 1], size=32)

            # Step 2: Encode watermark (low-level)
            signal_tensor = torch.FloatTensor(signal).to(device).unsqueeze(0)  # (1, T)
            payload_tensor = torch.FloatTensor(payload).to(device).unsqueeze(0)  # (1, 32)
            watermarked_tensor = model.encode(signal_tensor, payload_tensor)
            watermarked = watermarked_tensor.cpu().squeeze().numpy()

            # Save audio and payload
            sf.write(os.path.join(output_subdir, f"{base_filename}_original.wav"), signal, 16000)
            sf.write(os.path.join(output_subdir, f"{base_filename}_watermarked.wav"), watermarked, 16000)
            with open(os.path.join(output_subdir, f"{base_filename}_payload.txt"), "w") as f:
                f.write(f"Original Payload: {payload.tolist()}\n")

            # Step 3: Run all attacks
            for atk_name, _ in all_attacks():
                atk_tag = atk_name.replace("-", "")
                atk_tensor, _ = attack(torch.tensor(watermarked).unsqueeze(0).unsqueeze(0).to(torch.float32), 16000, [(atk_name, 1.0)])
                
                atk_tensor = restore_audio(atk_tensor, watermarked.shape[-1])
      
                atk_np = atk_tensor.squeeze().cpu().numpy()

                # Decode
                atk_input = torch.FloatTensor(atk_np).to(device).unsqueeze(0)
                decoded = model.decode(atk_input)

                if decoded is None:
                    decoded_bits = np.zeros_like(payload)
                    ber = 100.0
                else:
                    decoded_bits = (decoded >= 0.5).int().cpu().numpy().squeeze()
                    ber = (payload != decoded_bits).mean() * 100

                # Save attacked audio and result
                # sf.write(os.path.join(output_subdir, f"{base_filename}_attacked_{atk_tag}.wav"), atk_np, 16000)
                with open(os.path.join(output_subdir, f"{base_filename}_detect_{atk_tag}.txt"), "w") as f:
                    f.write(f"Decoded Payload: {decoded_bits.tolist()}\n")
                    f.write(f"Bit Error Rate: {ber:.1f}%\n")

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