import os, json, argparse
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write
from models import Generator, Encoder, Quantizer
from utils import load_checkpoint, AttrDict
from watermark import WatermarkEncoder, ImprovedWatermarkDecoder, random_watermark, attack, restore_audio, watermark_loss, count_common_digit
from meldataset import load_wav, mel_spectrogram, MAX_WAV_VALUE

# for each sample compute the average watermark loss under 10 random attacks
num_attack = 20

operations = ["CLP", "RSP-90", "Noise-W35", "SS-01", "AS-90", "AS-150",
                "EA-0301", "LP5000", "HP100", "MF-3", "TS-90", "TS-110"]

# Performs embedding of 0.5s audio exactly. For longer audio, clip them first.
# The inference includes: Encoder, RVQ, Decoder, WM encoder, WM decoder.
# Achieves watermark adding and extraction
@torch.no_grad()
def inference(a):
    # -------------------------------Loading modules and checkpoint -----------------------------------
    generator = Generator(h).to(device)
    encoder = Encoder(h).to(device)
    quantizer = Quantizer(h).to(device)
    watermark_encoder = WatermarkEncoder(h).to(device)
    watermark_decoder = ImprovedWatermarkDecoder(h).to(device)

    state_dict_codec = load_checkpoint(a.checkpoint_file, device)

    generator.load_state_dict(state_dict_codec["generator"])
    encoder.load_state_dict(state_dict_codec["encoder"])
    quantizer.load_state_dict(state_dict_codec["quantizer"])
    watermark_encoder.load_state_dict(state_dict_codec['watermark_encoder'])
    watermark_decoder.load_state_dict(state_dict_codec['watermark_decoder'])

    generator.eval()
    generator.remove_weight_norm()
    encoder.eval()
    encoder.remove_weight_norm()
    watermark_encoder.eval()
    watermark_decoder.eval()

    # ---------------------------- Testing for each sample audio ----------------------
    filelist = os.listdir(a.input_wavs_dir)
    for i, filename in enumerate(filelist):
        # --------------------- audio loading and chunking ------------------------------
        audio, sr = load_wav(os.path.join(a.input_wavs_dir, filename)) # sr: 24k
        audio /= MAX_WAV_VALUE
        audio = torch.FloatTensor(audio).to(device)

        # chunk the audio into 0.5s clips
        chunk_size = int(0.5 * sr)
        num_chunks = (len(audio) + chunk_size - 1) // chunk_size
        pad_len = num_chunks * chunk_size- len(audio)
        if pad_len > 0:
            audio = F.pad(audio, (0, pad_len))  # zero padding at the end
        audio_chunks = audio.view(num_chunks, chunk_size) # Reshape into chunks: (num_chunks, chunk_size): (num_chunks, 12000)

        # (num_chunks, 12000) -> (num_chunks, 1, 1, chunk_size)
        audio_chunks = audio_chunks.unsqueeze(1).unsqueeze(2)

        # ------------------- forward pass for each audio ---------------------
        generated_audio_chunks = []
        result_logs = []
        # process each chunk in the audio
        for _, audio_chunk in enumerate(audio_chunks):
            watermark = random_watermark(batch_size=1, h=h).to(device)
            watermark_feat = watermark_encoder(watermark)
            # (B, F, T): (1, 512, 50)
            imprinted_feat = encoder(audio_chunk, watermark_feat)
            # quantized_feat: (1, 50, 512) , loss: scalar, quantization_indexes: (16, 1, 1, 50)
            quantized_feat, loss_quantization, quantization_indexes = quantizer(imprinted_feat)
            # audio_generated: (1, 1, 12000)
            audio_generated = generator(quantized_feat)

            # save audio
            audio_generated = audio_generated.squeeze()
            generated_audio_chunks.append(audio_generated)

            # --------------------------------- each clip goes through random attack ------------------------
            # occurrence count, accuracy, loss
            result_log = {operation: [0, 0, 0] for operation in operations}
            # for each chunk, undergoes some attack to see average performance
            for j in range(num_attack):
                # audio_attacked: (32, 1, 12000)
                audio_attacked, attack_operation = attack(audio_generated, [
                    ("CLP", 0.35),
                    ("RSP-90", 0.15),
                    ("Noise-W35", 0.05),
                    ("SS-01", 0.05),
                    ("AS-90", 0.05), ("AS-150", 0.05),
                    ("EA-0301", 0.05),
                    ("LP5000", 0.05), ("HP100", 0.05), ("MF-3", 0.05),
                    ("TS-90", 0.05), ("TS-110", 0.05)
                ])
                # Keep audio length after time stretching attack for training. Clip or pad, this is no cheating.
                if attack_operation.startswith("TS"):
                    audio_attacked = restore_audio(audio_attacked, audio_generated.size(-1))
                # (1, 80, 50)
                mel_audio_attacked = mel_spectrogram(audio_attacked.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                     h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                # (1, 25)
                watermark_recovered = watermark_decoder(mel_audio_attacked)
                loss_watermark = watermark_loss(watermark_recovered, watermark)

                # log result
                result_log[attack_operation][0] += 1
                result_log[attack_operation][1] += count_common_digit(watermark, watermark_recovered)
                result_log[attack_operation][2] += loss_watermark

            for operation in operations:
                if result_log[operation][0] != 0:
                    # accuracy = total correct digit / (number of this attack * capacity in each time of attack)
                    result_log[operation][1] /= (result_log[operation][0] * h.Watermark["capacity"])
                    # average watermark loss for this kind of attack
                    result_log[operation[2]] /= result_log[operation][0]
                else:
                    # this attack never got picked
                    result_log[operation][1] = 0
                    result_log[operation][2] = 0

            # save the watermark test result for one audio clip
            result_logs.append(result_log)

        # --------------------------------- Saving and logging test result ----------------
        # Save all the watermarked chunks as one complete audio.
        full_generated_audio = torch.cat(generated_audio_chunks, dim=0)
        full_generated_audio *= MAX_WAV_VALUE
        full_generated_audio = full_generated_audio.cpu().numpy().astype('int16')
        output_file = os.path.join(a.output_dir, os.path.splitext(filename)[0] + '_generated.wav')
        write(output_file, sr, full_generated_audio)

        # attack count, accuracy, loss
        statistic = {operation: [0, 0, 0] for operation in operations}
        for result_dict in result_logs:
            for operation in operations:
                statistic[operation][0] += result_dict[operation][0]
                statistic[operation][1] += result_dict[operation][1]*result_dict[operation][0]
                statistic[operation][2] += result_dict[operation][2]*result_dict[operation][0]
        for operation, value in statistic.items():
            if value[0] != 0:
                statistic[operation] = [value[0], value[1]/value[0], value[2]/value[0]]
            else:
                statistic[operation] = [value[0], 0, 0]

        statistic_path = os.path.join(a.output_dir, 'statistics.json')
        with open(statistic_path, "w") as f:
            json.dump({f"{a.input_wavs_dir}/{filename} statistic": statistic}, f, indent=4)



device = 'cpu'
h = None
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_file', default='')
    parser.add_argument('--input_wavs_dir', default='')
    parser.add_argument('--output_dir', default='')
    a = parser.parse_args()

    global device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    global h
    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config.json')
    with open(config_file) as f:
        h = AttrDict(json.loads(f.read()))

    torch.manual_seed(h.seed)
    inference(a)

if __name__ == '__main__':
    main()
