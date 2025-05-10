# AudioSeal on LibriSpeech
#python AudioWatermark/inference_audioseal.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/AudioSeal/LibriSpeechOutputResult/
#python AudioWatermark/conclude_audioseal.py -r /root/autodl-tmp/AudioSeal/LibriSpeechOutputResult/
# WavMark on LibriSpeech
python AudioWatermark/inference_wavmark.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/WavMark/LibriSpeechOutputResult/
python AudioWatermark/conclude_wavmark.py -r /root/autodl-tmp/WavMark/LibriSpeechOutputResult/
# DCT on LibriSpeech
python AudioWatermark/inference_DCT.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/DCT/LibriSpeechOutputResult/
python AudioWatermark/conclude_DCT_and_patchwork.py -r /root/autodl-tmp/DCT/LibriSpeechOutputResult/
# Patchwork on LibriSpeech
python AudioWatermark/inference_patchwork.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/Patchwork/LibriSpeechOutputResult/
python AudioWatermark/conclude_DCT_and_patchwork.py -r /root/autodl-tmp/Patchwork/LibriSpeechOutputResult/

# AudioSeal on Voxpopuli
python AudioWatermark/inference_audioseal.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/AudioSeal/VoxpopuliOutputResult/
python AudioWatermark/conclude_audioseal.py -r /root/autodl-tmp/AudioSeal/VoxpopuliOutputResult/
# WavMark on Voxpopuli
python AudioWatermark/inference_wavmark.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/WavMark/VoxpopuliOutputResult/
python AudioWatermark/conclude_wavmark.py -r /root/autodl-tmp/WavMark/VoxpopuliOutputResult/
# DCT on Voxpopuli
python AudioWatermark/inference_DCT.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/DCT/VoxpopuliOutputResult/
python AudioWatermark/conclude_DCT_and_patchwork.py -r /root/autodl-tmp/DCT/VoxpopuliOutputResult/
# Patchwork on Voxpopuli
python AudioWatermark/inference_patchworkpy -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/Patchwork/VoxpopuliOutputResult/
python AudioWatermark/conclude_DCT_and_patchwork.py -r /root/autodl-tmp/Patchwork/VoxpopuliOutputResult/


# AudioSeal puts image on LibriSpeech
python ImageWatermark/inference_audioseal_img.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/AudioSeal/LibriSpeechOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/AudioSeal/LibriSpeechOutputResult_Img/
# WavMark puts image on LibriSpeech
python ImageWatermark/inference_wavmark_img.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/WavMark/LibriSpeechOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/WavMark/LibriSpeechOutputResult_Img/
# DCT puts image on LibriSpeech
python ImageWatermark/inference_DCT_img.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/DCT/LibriSpeechOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/DCT/LibriSpeechOutputResult_Img/
# Patchwork puts image on LibriSpeech
python ImageWatermark/inference_patchwork_img.py -f /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/librispeec_val_filelist_16k.txt -d /root/autodl-tmp/LibriSpeech/LibriSpeech_16k_wav/dev-clean -o /root/autodl-tmp/Patchwork/LibriSpeechOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/Patchwork/LibriSpeechOutputResult_Img/

# AudioSeal puts image on Voxpopuli
python ImageWatermark/inference_audioseal_img.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/AudioSeal/VoxpopuliOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/AudioSeal/VoxpopuliOutputResult_Img/
# WavMark puts image on Voxpopuli
python ImageWatermark/inference_wavmark_img.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/WavMark/VoxpopuliOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/WavMark/VoxpopuliOutputResult_Img/
# DCT puts image on Voxpopuli
python ImageWatermark/inference_DCT_img.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/DCT/VoxpopuliOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/DCT/VoxpopuliOutputResult_Img/
# Patchwork puts image on Voxpopuli
python ImageWatermark/inference_patchwork_img.py -f /root/autodl-tmp/voxpopuli_dataset/val_16k/voxpopuli_val_filelist_16k_9s_5000.txt -d /root/autodl-tmp/voxpopuli_dataset/val_16k/val_voxpopuli_16k_wav -o /root/autodl-tmp/Patchwork/VoxpopuliOutputResult_Img/
python ImageWatermark/conclude_model_img.py -r /root/autodl-tmp/Patchwork/VoxpopuliOutputResult_Img/

