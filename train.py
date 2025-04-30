'''
#  Start training:
python train.py --input_training_file train_filelist.txt --input_validation_file val_filelist.txt --checkpoint_path ~/autodl-tmp/WMCapstone/ckpt --pretrain_checkpoint_path /root/autodl-tmp/pretrain_ckpt/ --log_path ~/tf-logs

# Resume training
python train.py --input_training_file train_filelist.txt --input_validation_file val_filelist.txt --checkpoint_path ~/autodl-tmp/WMCapstone/ckpt --log_path ~/tf-logs

'''

import torch
import os
import itertools
import time
import json
import argparse
import torch.nn.functional as func
import torch.multiprocessing as mp

from models import Encoder, Quantizer, Generator, feature_loss, generator_loss
from watermark import random_watermark, WatermarkEncoder, ImprovedWatermarkDecoder, attack, restore_audio, watermark_loss
from discriminators import MultiScaleDiscriminator, MultiPeriodDiscriminator, MultiScaleSTFTDiscriminator, discriminator_loss
from dataset import get_dataset_filelist, MelDataset, mel_spectrogram
from utils import scan_checkpoint, load_checkpoint, save_checkpoint, AttrDict, build_env, plot_spectrogram

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group

plot_audio_number = 4

# rank: process id, a: input parameters, h: hyperparameter settings
def train(rank, a, h):
    # ------------------------- Load models and ckpt on multi GPUs --------------------------- *
    if h.num_gpus > 1:
        # torch.distributed module. It initializes the default process group for distributed training
        # backend: communication backend to use (nccl for nvidia); init_method: Specifies how the processes initialize their connection.
        # world_size: Total number of processes in a GPU. rank=The unique ID of the current process within the world_size.
        init_process_group(backend=h.dist_config['dist_backend'], init_method=h.dist_config['dist_url'],
                           world_size=h.dist_config['world_size'] * h.num_gpus, rank=rank)
    torch.cuda.manual_seed(h.seed)
    device = torch.device('cuda:{:d}'.format(rank))

    ## load models
    encoder = Encoder(h).to(device)
    generator = Generator(h).to(device)
    quantizer = Quantizer(h).to(device)
    watermark_encoder = WatermarkEncoder(h).to(device)
    watermark_decoder = ImprovedWatermarkDecoder(h).to(device)
    mp_discriminator = MultiPeriodDiscriminator().to(device)
    ms_discriminator = MultiScaleDiscriminator().to(device)
    msstft_discriminator = MultiScaleSTFTDiscriminator(32).to(device)

    if rank == 0:
        print(encoder)
        print(quantizer)
        print(generator)
        print(watermark_encoder)
        print(watermark_decoder)
        os.makedirs(a.checkpoint_path, exist_ok=True)
        print("checkpoints directory : ", a.checkpoint_path)

    ## load checkpoint to resume training
    ckpt_codec_path = ckpt_discriminators_optimizer_path = None
    steps = 0
    state_dict_discrim_and_optim = None
    state_dict_codec = None
    last_epoch = -1
    if os.path.isdir(a.pretrain_checkpoint_path):
        # load from pretrain model: TO DO THIS ENSURE SAME SETTING AS HIFI-CODEC.
        ckpt_codec_path = scan_checkpoint(a.pretrain_checkpoint_path, 'HiFi-Codec')
        if ckpt_codec_path is not None:
            state_dict_codec = load_checkpoint(ckpt_codec_path, device)

            # load generator: generator is the same as HiFiCodec
            generator.load_state_dict(state_dict_codec['generator'])

            # load encoder
            encoder_state_dict = encoder.state_dict()
            match_state_dict = {}
            for k, v in state_dict_codec['encoder'].items():
                # load only if they have the same name and shape
                if k in encoder_state_dict and encoder_state_dict[k].shape == v.shape:
                    match_state_dict[k] = v
            encoder_state_dict.update(match_state_dict)
            encoder.load_state_dict(encoder_state_dict)

            # load quantizer
            quantizer_state_dict = quantizer.state_dict() # returns reference, not a deep copy
            for residual_layer_idx in range(2):
                old_prefix = f"quantizer_modules{'' if residual_layer_idx == 0 else '2'}"
                new_prefix = f"quantizer_module_residual_list.{residual_layer_idx}"

                for code_group_idx in range(quantizer.n_code_groups): # 2
                    old_key = f"{old_prefix}.{code_group_idx}.embedding.weight" #quantizer_model2.1.embedding.weight
                    new_key = f"{new_prefix}.{code_group_idx}.embedding.weight" # quantizer_module_residual_list.1.1.embedding.weight

                    if old_key in state_dict_codec['quantizer'] and new_key in quantizer_state_dict:
                        quantizer_state_dict[new_key].copy_(state_dict_codec['quantizer'][old_key])
    else:
        # load from previously saved result.
        if os.path.isdir(a.checkpoint_path):
            ckpt_codec_path = scan_checkpoint(a.checkpoint_path, 'generator_')
            ckpt_discriminators_optimizer_path = scan_checkpoint(a.checkpoint_path, 'discriminators_and_optimizer_')


        if ckpt_codec_path is not None and ckpt_discriminators_optimizer_path is not None:
            state_dict_codec = load_checkpoint(ckpt_codec_path, device)
            state_dict_discrim_and_optim = load_checkpoint(ckpt_discriminators_optimizer_path, device)

            # encoder-RVQ-decoder loading
            generator.load_state_dict(state_dict_codec['generator'])
            encoder.load_state_dict(state_dict_codec['encoder'])
            quantizer.load_state_dict(state_dict_codec['quantizer'])
            # watermark_encoder.load_state_dict(state_dict_codec['watermark_encoder'])
            # watermark_decoder.load_state_dict(state_dict_codec['watermark_decoder'])

            # discriminator and optimizer loading
            mp_discriminator.load_state_dict(state_dict_discrim_and_optim['mpd'])
            ms_discriminator.load_state_dict(state_dict_discrim_and_optim['msd'])
            msstft_discriminator.load_state_dict(state_dict_discrim_and_optim['msstftd'])

            # other loading
            steps = state_dict_discrim_and_optim['steps'] + 1
            last_epoch = state_dict_discrim_and_optim['epoch']


    # ----------------------- Handling training and validation data distribution across GPUs ------------- *
    if h.num_gpus > 1:
        generator = DistributedDataParallel(generator, device_ids=[rank]).to(device)
        encoder = DistributedDataParallel(encoder, device_ids=[rank]).to(device)
        quantizer = DistributedDataParallel(quantizer, device_ids=[rank]).to(device)
        watermark_encoder = DistributedDataParallel(watermark_encoder, device_ids=[rank]).to(device)
        watermark_decoder = DistributedDataParallel(watermark_decoder, device_ids=[rank]).to(device)
        mp_discriminator = DistributedDataParallel(mp_discriminator, device_ids=[rank]).to(device)
        ms_discriminator = DistributedDataParallel(ms_discriminator, device_ids=[rank]).to(device)
        msstft_discriminator = DistributedDataParallel(msstft_discriminator, device_ids=[rank]).to(device)

    training_filelist, validation_filelist = get_dataset_filelist(a)

    train_set = MelDataset(training_filelist, h.segment_size, h.n_fft, h.num_mels,
                          h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, n_cache_reuse=0,
                          shuffle=False if h.num_gpus > 1 else True, fmax_loss=h.fmax_for_loss, device=device,
                          fine_tuning=a.fine_tuning, base_mels_path=a.input_mels_dir)
    # DistributedSampler is for multi-GPUs
    train_sampler = DistributedSampler(train_set) if h.num_gpus > 1 else None
    # num_workers: how many subprocesses to use for data loading.
    train_loader = DataLoader(train_set, num_workers=h.num_workers, shuffle=False,
                              sampler=train_sampler,
                              batch_size=h.batch_size,
                              pin_memory=True,
                              drop_last=True)

    if rank == 0:
        valid_set = MelDataset(validation_filelist, h.segment_size, h.n_fft, h.num_mels,
                              h.hop_size, h.win_size, h.sampling_rate, h.fmin, h.fmax, False, False, n_cache_reuse=0,
                              fmax_loss=h.fmax_for_loss, device=device, fine_tuning=a.fine_tuning,
                              base_mels_path=a.input_mels_dir)
        validation_loader = DataLoader(valid_set, num_workers=1, shuffle=False,
                                       sampler=None,
                                       batch_size=1,
                                       pin_memory=True,
                                       drop_last=True)
        summary_writer = SummaryWriter(os.path.join(a.log_path, 'logs'))

    #------------------------- Configuring optimizer and scheduler and freeze weight training -------------------*
    # Freeze weight for faster training
    
    trainable_params_codec = filter(lambda p: p.requires_grad,
                              itertools.chain(
                                  generator.parameters(),
                                  encoder.parameters(),
                                  quantizer.parameters(),
                                  watermark_encoder.parameters(),
                                  watermark_decoder.parameters()))
    optim_codec = torch.optim.Adam(trainable_params_codec, h.learning_rate, betas=(h.adam_b1, h.adam_b2))

    
    trainable_params_discriminators = filter(lambda p: p.requires_grad,
                                             itertools.chain(ms_discriminator.parameters(),
                                                             mp_discriminator.parameters(),
                                                             msstft_discriminator.parameters()))
    optim_discriminators = torch.optim.Adam(trainable_params_discriminators, h.learning_rate, betas=(h.adam_b1, h.adam_b2))

    # load optimizer state if resume training
    if state_dict_discrim_and_optim is not None:
        # optim_codec.load_state_dict(state_dict_discrim_and_optim['optim_codec'])
        optim_discriminators.load_state_dict(state_dict_discrim_and_optim['optim_discriminators'])
        
    # for stage 2+ training
    '''
    for name, param in encoder.named_parameters():
        if "AIU" not in name:
            param.requires_grad = False
    '''
    '''
    for name, param in encoder.named_parameters():
        if "AIU" in name:
            param.requires_grad = False
    '''
    '''
    # for stage 2+ training
    # encoder
    for param in encoder.parameters():
        param.requries_grad = False
    '''
    '''
    # decoder
    for param in generator.parameters():
        param.requires_grad = False
    # quantizer
    for param in quantizer.parameters():
        param.requires_grad = False

    # discriminators
    for param in trainable_params_discriminators:
        param.requires_grad = False
    '''
    '''
    # Manual LR adjust during training
    for param_group in optim_codec.param_groups:
        param_group['lr'] = param_group['lr'] / 3  # or set an exact value
    '''   
    '''
    scheduler_codec = torch.optim.lr_scheduler.ExponentialLR(optim_codec, gamma=h.lr_decay, last_epoch=last_epoch)
    scheduler_discriminators = torch.optim.lr_scheduler.ExponentialLR(optim_discriminators, gamma=h.lr_decay, last_epoch=last_epoch)
    '''
    
    scheduler_codec = torch.optim.lr_scheduler.ExponentialLR(optim_codec, gamma=h.lr_decay, last_epoch=-1)
    scheduler_discriminators = torch.optim.lr_scheduler.ExponentialLR(optim_discriminators, gamma=h.lr_decay, last_epoch=-1)
    
    #---------------------------- Set models to Train mode -----------------------*
    generator.train()
    encoder.train()
    quantizer.train()
    watermark_encoder.train()
    watermark_decoder.train()
    mp_discriminator.train()
    ms_discriminator.train()
    # msstft_discriminator in the original code is not trained

    #--------------------------- Training Stage -----------------------------------*
    for epoch in tqdm(range(max(0, last_epoch), a.training_epochs), desc="Epoch Progress", unit="epoch"):
        if h.num_gpus > 1:
            train_sampler.set_epoch(epoch)
        # logging epoch train time
        if rank == 0:
            start_e = time.time()
            print("Epoch: {}".format(epoch+1))

        # --------------------------batch training------------------------------------*
        for i, batch in tqdm(enumerate(train_loader), desc=f"Batch Progress (Epoch {epoch + 1})", unit="batch", total=len(train_loader)):
            # logging batch train time
            if rank == 0:
                start_b = time.time()

            # --------------------Loading data sample and pass to GPU memory--------*
            # mel: (32, 80, 50); audio: (32, 12000)
            mel_audio, audio, _, mel_audio_loss = batch
            watermark = random_watermark(h.batch_size, h)

            mel_audio = mel_audio.to(device, non_blocking=True)
            audio = audio.to(device, non_blocking=True)
            mel_audio_loss = mel_audio_loss.to(device, non_blocking=True)
            watermark = watermark.to(device, non_blocking=True)

            # -------------------- Forward pass for codec --------------------*
            # (32, 50, 512)
            watermark_feat = watermark_encoder(watermark)
            # (32, 1, 12000)
            audio = audio.unsqueeze(1)
            # (B, F, T): (32, 512, 50)
            imprinted_feat = encoder(audio, watermark_feat)
            # quantized_feat: (32, 50, 512) , loss: scalar, quantization_indexes: (16, 1, 32, 50)
            quantized_feat, loss_quantization, quantization_indexes = quantizer(imprinted_feat)
            # audio_generated: (32, 1, 12000)
            audio_generated = generator(quantized_feat)
            # audio_attacked: (32, 1, 12000)
            
            audio_attacked, attack_operation = attack(audio_generated, h.sampling_rate, [
                ("Pass", 0.35),
                ("RSP-70", 0.15),
                ("Noise-W55", 0.05),
                ("SS-01", 0.05),
                ("AS-50", 0.05), ("AS-250", 0.05),
                ("EA-0301", 0.05),
                ("LP1000", 0.05), ("HP500", 0.05), ("MF-6", 0.05),
                ("TS-90", 0.05), ("TS-110", 0.05)
            ])
            # Keep audio length after time stretching attack for training
            audio_attacked = restore_audio(audio_attacked, audio_generated.size(-1))

            '''
            # for stage 1 training, no attack
            audio_attacked = audio_generated
            '''
            
            # ------------------ watermark codec forward & loss compute ------------------------*
            # (32, 25)
            mel_audio_attacked = mel_spectrogram(audio_attacked.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate,
                                                 h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
            rec_watermark_logit, watermark_recovered = watermark_decoder(mel_audio_attacked)
            loss_watermark = watermark_loss(rec_watermark_logit, watermark)

            # ---------------------------- Forward pass for discriminators -----------------------*
            # TODO: in the original code, the generated input is audio_attacked. I think it is a mistake
            labels_origin_audio_mpd, labels_generated_audio_mpd, _, _ = mp_discriminator(audio, audio_generated.detach())
            labels_origin_audio_msd, labels_generated_audio_msd, _, _ = ms_discriminator(audio, audio_generated.detach())
            labels_origin_audio_msstftd, _ = msstft_discriminator(audio)
            labels_generated_audio_msstftd, _ = msstft_discriminator(audio_generated.detach())

            # ----------- Discriminator loss computation, backward prob, gradient updates--------*
            loss_mpd, _, _ = discriminator_loss(labels_origin_audio_mpd, labels_generated_audio_mpd)
            loss_msd, _, _ = discriminator_loss(labels_origin_audio_msd, labels_generated_audio_msd)
            loss_msstftd, _, _ = discriminator_loss(labels_origin_audio_msstftd, labels_generated_audio_msstftd)

            loss_discriminators = loss_mpd + loss_msd + loss_msstftd

            # backward and gradient update
            
            optim_discriminators.zero_grad()
            loss_discriminators.backward()
            optim_discriminators.step()

            # -------------------- audio codec loss: mel spectrogram loss -------------------------------*
            # the generated mel spectrogram should be similar to the original one.

            # get more mel spectrogram with different setting to construct better loss function
            # (32, 80, 50), n_fft=1024, hop size=240 (12000/240=50), #mel filters=80. Note that n_fft has no effect on the mel output dim.
            # TODO: in the original code, the generated input is audio_attacked. I think it is a mistake
            mel_audio_generated = mel_spectrogram(audio_generated.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
            # mel_audio = mel_spectrogram(audio.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss) # computed when loading dataset.
            loss_mel_base = func.l1_loss(mel_audio_loss, mel_audio_generated)

            # (32, 80, 100), n_fft=512, hop size=120 (12000/120=100), #mel filters=80
            mel_audio_generated_1 = mel_spectrogram(audio_generated.squeeze(1), 512, h.num_mels, h.sampling_rate, 120, 512, h.fmin, h.fmax_for_loss)
            mel_audio_1 = mel_spectrogram(audio.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, 120, 512, h.fmin, h.fmax_for_loss)
            loss_mel_1 = func.l1_loss(mel_audio_1, mel_audio_generated_1)

            # (32, 80, 200), n_fft=256, hop size=60 (12000/60=200), #mel filters=80
            mel_audio_generated_2 = mel_spectrogram(audio_generated.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin, h.fmax_for_loss)
            mel_audio_2 = mel_spectrogram(audio.squeeze(1), 256, h.num_mels, h.sampling_rate, 60, 256, h.fmin, h.fmax_for_loss)
            loss_mel_2 = func.l1_loss(mel_audio_2, mel_audio_generated_2)

            # total mel spectrogram loss
            loss_mel_total = loss_mel_base * 45 + loss_mel_1 + loss_mel_2

            # ------------------- audio codec loss: feature map loss ---------------------------*
            # The generator is encouraged to replicate the hierarchical feature maps of real audio.
            # The feature maps comes from the discriminators. I.e. to fool the discriminators

            # the discriminators' output is recalculated to show the gradient updates in discriminators.
            # TODO: in the original code, the generated input is audio_attacked. I think it is a mistake
            labels_origin_audio_mpd, labels_generated_audio_mpd, fmap_origin_audio_mpd, fmap_generated_audio_mpd = mp_discriminator(audio, audio_generated.detach())
            labels_origin_audio_msd, labels_generated_audio_msd, fmap_origin_audio_msd, fmap_generated_audio_msd = ms_discriminator(audio, audio_generated.detach())
            labels_origin_audio_msstftd, fmap_origin_audio_msstftd = msstft_discriminator(audio)
            labels_generated_audio_msstftd, fmap_generated_audio_msstftd = msstft_discriminator(audio_generated.detach())

            loss_fmap_mpd = feature_loss(fmap_origin_audio_mpd, fmap_generated_audio_mpd)
            loss_fmap_msd = feature_loss(fmap_origin_audio_msd, fmap_generated_audio_msd)
            loss_fmap_msstftd = feature_loss(fmap_origin_audio_msstftd, fmap_generated_audio_msstftd)

            loss_fmap_total = loss_fmap_mpd + loss_fmap_msd + loss_fmap_msstftd

            # ------------------ audio codec loss: label loss -----------------------------------*
            # Let discriminators confuse the original and the generated by having them assign
            # label 'real' to generated audio.

            loss_label_mpd, _ = generator_loss(labels_generated_audio_mpd)
            loss_label_msd, _ = generator_loss(labels_generated_audio_msd)
            loss_label_msstftd, _ = generator_loss(labels_generated_audio_msstftd)

            loss_label_total = loss_label_mpd + loss_label_msd + loss_label_msstftd

            # ----------------- Codec total loss and gradient update ------------------------------*
            loss_codec_total = loss_quantization * 10 + loss_mel_total + loss_fmap_total + loss_label_total + loss_watermark * 5
            # loss_codec_total = loss_quantization * 0 + loss_mel_total * 0 + loss_fmap_total * 0 + loss_label_total * 0 + loss_watermark * 5

            optim_codec.zero_grad()
            loss_codec_total.backward()
            optim_codec.step()

            # ------------------------ Logging and saving ---------------------*
            if rank == 0:
                # STDOUT logging
                if steps % a.stdout_interval == 0:
                    with torch.no_grad():
                        print('Steps : {:d}, Codec Loss Total : {:4.3f}, Quantizer Loss : {:4.3f}, Mel-Spec. Error : {:4.3f}, watermark_loss : {:4.3f}, s/b : {:4.3f}'.
                              format(steps, loss_codec_total, loss_quantization, loss_mel_base.item(), loss_watermark, time.time() - start_b))
                # checkpointing
                if steps % a.checkpoint_interval == 0 and steps != 0:
                    print("check pointing")
                    checkpoint_path = "{}/generator_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'generator': (generator.module if h.num_gpus > 1 else generator).state_dict(),
                                     'encoder': (encoder.module if h.num_gpus > 1 else encoder).state_dict(),
                                     'quantizer': (quantizer.module if h.num_gpus > 1 else quantizer).state_dict(),
                                     'watermark_encoder': (watermark_encoder.module if h.num_gpus > 1 else watermark_encoder).state_dict(),
                                     'watermark_decoder': (watermark_decoder.module if h.num_gpus > 1 else watermark_decoder).state_dict()
                                     }, num_ckpt_keep=a.num_ckpt_keep)
                    checkpoint_path = "{}/discriminators_and_optimizer_{:08d}".format(a.checkpoint_path, steps)
                    save_checkpoint(checkpoint_path,
                                    {'mpd': (mp_discriminator.module if h.num_gpus > 1
                                                         else mp_discriminator).state_dict(),
                                     'msd': (ms_discriminator.module if h.num_gpus > 1
                                                         else ms_discriminator).state_dict(),
                                     'msstftd': (msstft_discriminator.module if h.num_gpus > 1
                                                               else msstft_discriminator).state_dict(),
                                     'optim_codec': optim_codec.state_dict(),
                                     'optim_discriminators': optim_discriminators.state_dict(),
                                     'steps': steps,
                                     'epoch': epoch}, num_ckpt_keep=a.num_ckpt_keep)
                # tensorboard logging:
                if steps % a.summary_interval == 0:
                    summary_writer.add_scalar("training/codec_loss_total", loss_codec_total, steps)
                    summary_writer.add_scalar("training/mel_spec_error", loss_mel_base.item(), steps)

                torch.cuda.empty_cache()

            # ------------------------------- validation --------------------------------*
            if rank == 0 and steps % a.validation_interval == 0 and steps != 0:
                print("validating")
                encoder.eval()
                quantizer.eval()
                generator.eval()
                watermark_encoder.eval()
                watermark_decoder.eval()

                ##  --------------------------batch validation ----------------------
                loss_validation_mel_audio = 0
                loss_validation_watermark = 0

                with torch.no_grad():
                    for index, batch in enumerate(validation_loader):
                        ##  ---------Loading data sample and pass to GPU memory
                        # mel: (1, 80, 50); audio: (1, 12000), batch size = 1
                        mel_audio, audio, _, mel_audio_loss = batch
                        # (32, 25)
                        watermark = random_watermark(mel_audio.shape[0], h)

                        audio = audio.to(device)
                        mel_audio_loss = mel_audio_loss.to(device)
                        watermark = watermark.to(device)

                        ## -------- Codec forward pass and loss compute
                        # (1, 50, 512)
                        watermark_feat = watermark_encoder(watermark)
                        audio = audio.unsqueeze(1)
                        # (B, F, T): (1, 512, 50)
                        imprinted_feat = encoder(audio, watermark_feat)
                        # quantized_feat: (1, 50, 512) , loss: scalar, quantization_indexes: (16, 1, 1, 50)
                        quantized_feat, loss_quantization, quantization_indexes = quantizer(imprinted_feat)
                        # audio_generated: (1, 1, 12000)
                        audio_generated = generator(quantized_feat)
                        # audio_attacked: (1, 1, 12000)
                        audio_attacked, attack_operation = attack(audio_generated, h.sampling_rate, [
                            ("Pass", 0.35),
                            ("RSP-70", 0.15),
                            ("Noise-W55", 0.05),
                            ("SS-01", 0.05),
                            ("AS-50", 0.05), ("AS-250", 0.05),
                            ("EA-0301", 0.05),
                            ("LP1000", 0.05), ("HP500", 0.05), ("MF-6", 0.05),
                            ("TS-90", 0.05), ("TS-110", 0.05)
                        ])
                        # Keep audio length after time stretching attack for training
                        audio_attacked = restore_audio(audio_attacked, audio_generated.size(-1))

                        # -------- audio codec loss computation
                        mel_audio_generated = mel_spectrogram(audio_generated.squeeze(1), h.n_fft, h.num_mels, h.sampling_rate, h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        loss_validation_mel_audio += func.l1_loss(mel_audio_loss, mel_audio_generated).item()

                        ##-------- watermark codec forward & loss compute
                        mel_audio_attacked = mel_spectrogram(audio_attacked.squeeze(1), h.n_fft, h.num_mels,
                                                             h.sampling_rate,
                                                             h.hop_size, h.win_size, h.fmin, h.fmax_for_loss)
                        rec_watermark_logit, watermark_recovered = watermark_decoder(mel_audio_attacked)
                        loss_validation_watermark += watermark_loss(rec_watermark_logit, watermark)

                        ## ---------------- plot the original audio and the generated audio and wm
                        global plot_ground_truth_only_once, plot_audio_number
                        # log only the first few in the validation set.
                        if index <= plot_audio_number:
                            # ground truth
                            summary_writer.add_audio('gt/audio_{}'.format(index), audio[0], steps, h.sampling_rate)
                            summary_writer.add_figure('gt/audio_mel_spec_{}'.format(index), plot_spectrogram(mel_audio[0]), steps)

                            # generated
                            summary_writer.add_audio('generated/audio_generated_{}'.format(index), audio_generated[0], steps, h.sampling_rate)
                            summary_writer.add_figure('generated/mel_audio_generated_{}'.format(index), plot_spectrogram(mel_audio_generated.squeeze(0).cpu().numpy()), steps)

                            # convert watermark index to character in ASCII
                            watermark_words = "".join(chr(idx+65) for idx in watermark.view(-1).tolist())
                            watermark_words_recovered = "".join(chr(idx+65) for idx in watermark_recovered.view(-1).tolist())
                            summary_writer.add_text("original_watermark", watermark_words, steps)
                            summary_writer.add_text("recovered_watermark", watermark_words_recovered, steps)


                    # ------------------ loss compute for all validation samples --------*
                    loss_validation_watermark /= (index+1)
                    loss_validation_mel_audio /= (index+1)
                    summary_writer.add_scalar("validation/mel_spec_error", loss_validation_mel_audio, steps)
                    summary_writer.add_scalar("validation/watermark_error", loss_validation_watermark, steps)

                # ---------- last line for validation---------------
                # TODO: only the generator and the watermark_decoder are set back to train mode in original code
                generator.train()
                encoder.train()
                quantizer.train()
                watermark_encoder.train()
                watermark_decoder.train()

            # ------------- last line for this batch--------------- -------------------
            steps += 1

        # ------------------ last line of this epoch:  one epoch over. --------------------------
        
        scheduler_discriminators.step()
        
        scheduler_codec.step()
        if rank == 0:
            print('Time taken for epoch {} is {} sec\n'.format(epoch + 1, int(time.time() - start_e)))

    # ----------------------- last line of train function: Train over.-------------------------
    print("Training is over. ")

def main():
    print('Initializing Training Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_training_file', default='')
    parser.add_argument('--input_validation_file', default='')
    parser.add_argument('--checkpoint_path', default='')
    parser.add_argument('--config', default='./config.json')
    parser.add_argument('--training_epochs', default=2000, type=int)
    parser.add_argument('--stdout_interval', default=5, type=int) # 5
    parser.add_argument('--checkpoint_interval', default=2000, type=int) # 20 5000
    parser.add_argument('--summary_interval', default=100, type=int) # 100
    parser.add_argument('--log_path', default='')
    parser.add_argument('--validation_interval', default=2000, type=int) # 20 1000 500
    parser.add_argument('--num_ckpt_keep', default=5, type=int) # 5
    parser.add_argument('--fine_tuning', default=False, type=bool)
    parser.add_argument('--input_mels_dir', default='')
    parser.add_argument('--pretrain_checkpoint_path', default='')

    a = parser.parse_args()

    with open(a.config) as f:
        data = f.read()

    json_config = json.loads(data)
    h = AttrDict(json_config)
    build_env(a.config, 'config.json', a.checkpoint_path)

    torch.manual_seed(h.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        h.num_gpus = torch.cuda.device_count()
        h.batch_size = int(h.batch_size / h.num_gpus)
        print('Batch size per GPU :', h.batch_size)
    else:
        pass

    if h.num_gpus > 1:
        mp.spawn(train, nprocs=h.num_gpus, args=(a, h,))
    else:
        train(0, a, h)


if __name__ == '__main__':
    main()