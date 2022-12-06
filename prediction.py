import argparse
import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import librosa
import yaml
from tqdm import tqdm

from tensorflow_tts.configs import MultiBandMelGANGeneratorConfig, MelGANGeneratorConfig
from tensorflow_tts.models import TFPQMF, TFMelGANGenerator

import models

def preprocess(wav_path, hifi = False):
    # load config
    with open("./vocoder/ljspeech_preprocess.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    if hifi:
        config["sampling_rate"] = 44100
        config["hop_size"] = 512
        config["fmin"] = 20
        config["fmax"] = 11025
        config["fft_size"] = 2048
        
    audio, rate = sf.read(wav_path)
    audio = audio.astype(np.float32)

    # check sample rate
    if rate != config["sampling_rate"]:
        audio = librosa.resample(audio, rate, config["sampling_rate"])

    # trim silence
    if config["trim_silence"]:
        audio, _ = librosa.effects.trim(
                audio,
                top_db=config["trim_threshold_in_db"],
                frame_length=config["trim_frame_size"],
                hop_length=config["trim_hop_size"],
            )
    
    sampling_rate = config["sampling_rate"]
    hop_size = config["hop_size"]

    # get spectrogram
    D = librosa.stft(
        audio,
        n_fft=config["fft_size"],
        hop_length=hop_size,
        win_length=config["win_length"],
        window=config["window"],
        pad_mode="reflect",
    )
    S, _ = librosa.magphase(D)  # (#bins, #frames)

    # get mel basis
    fmin = 0 if config["fmin"] is None else config["fmin"]
    fmax = sampling_rate // 2 if config["fmax"] is None else config["fmax"]
    mel_basis = librosa.filters.mel(
        sr=sampling_rate,
        n_fft=config["fft_size"],
        n_mels=config["num_mels"],
        fmin=fmin,
        fmax=fmax,
    )
    mel = np.log10(np.maximum(np.dot(mel_basis, S), 1e-10)).T  # (#frames, #bins)

    scaler = StandardScaler()
    scaler.n_features_in_ = config["num_mels"]
    
    mel_norm = scaler.fit_transform(mel)
    
    
    # check audio and feature length
    audio = np.pad(audio, (0, config["fft_size"]), mode="edge")
    audio = audio[: len(mel) * hop_size]
    assert len(mel) * hop_size == len(audio)
    
    return mel_norm
    
   
def main_multiband_hf():
    out_dir = "./prediction/multiband_melgan_hf.v1/"
    # check directory existence
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load config
    with open("./vocoder/multiband_melgan_hf/conf/multiband_melgan_hf.lju.v1.yml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    mel = preprocess("./datasets/jvs_datasets/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav", True)
    # mel = preprocess("/workspace/StarGAN-VC2-tf/datasets/vcc2018_datasets/vcc2018_training/VCC2TF1/10001.wav")

    t = mel.shape[0]
    t = t-t%4
    mel = mel.T[np.newaxis, :, :t, np.newaxis]

    converter = models.Generator()
    converter(tf.random.uniform((1, 80, 64, 2)))
    converter.load_weights("/workspace/MaskCycleGAN-VC-tf/saved_models/maskcyclegan_vc2/20221205-045950/62500/X2Y.h5")
    mel = converter(np.concatenate([mel, np.ones(mel.shape)], axis = -1), training=False)

    mel = np.squeeze(mel).T[np.newaxis, :, :]

    # define model and load checkpoint
    mb_melgan = TFMelGANGenerator(config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),name="multiband_melgan_generator",)
    mb_melgan._build()
    mb_melgan.load_weights("./vocoder/multiband_melgan_hf/checkpoints/generator-920000.h5")

    pqmf = TFPQMF(config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf")

    # melgan inference.
    generated_subbands = mb_melgan(mel)
    generated_audios = pqmf.synthesis(generated_subbands)

    # convert to numpy.
    generated_audios = generated_audios.numpy()  # [B, T]

    # save to outdir
    for i, audio in enumerate(generated_audios):
        sf.write(
            os.path.join(out_dir, f"jvs002.wav"),
            audio[: t * config["hop_size"]],
            config["sampling_rate"],
            "PCM_16",
        )
 
    
def main_multiband():
    out_dir = "./prediction/multiband_melgan.v1/"
    # check directory existence
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load config
    with open("./vocoder/multiband_melgan/conf/multiband_melgan.v1.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    mel = preprocess("./datasets/jvs_datasets/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav")
    # mel = preprocess("/workspace/StarGAN-VC2-tf/datasets/vcc2018_datasets/vcc2018_training/VCC2TF1/10001.wav")

    t = mel.shape[0]
    t = t-t%4
    mel = mel.T[np.newaxis, :, :t, np.newaxis]

    converter = models.Generator()
    converter(tf.random.uniform((1, 80, 64, 2)))
    converter.load_weights("/workspace/MaskCycleGAN-VC-tf/saved_models/maskcyclegan_vc2/20221205-045950/62500/X2Y.h5")
    # mel = converter(np.concatenate([mel, np.ones(mel.shape)], axis = -1), training=False)

    mel = np.squeeze(mel).T[np.newaxis, :, :]

    # define model and load checkpoint
    mb_melgan = TFMelGANGenerator(config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]),name="multiband_melgan_generator",)
    mb_melgan._build()
    mb_melgan.load_weights("./vocoder/multiband_melgan/checkpoints/generator-940000.h5")

    pqmf = TFPQMF(config=MultiBandMelGANGeneratorConfig(**config["multiband_melgan_generator_params"]), name="pqmf")



    # melgan inference.
    generated_subbands = mb_melgan(mel)
    generated_audios = pqmf.synthesis(generated_subbands)

    # convert to numpy.
    generated_audios = generated_audios.numpy()  # [B, T]

    # save to outdir
    for i, audio in enumerate(generated_audios):
        sf.write(
            os.path.join(out_dir, f"jvs002.wav"),
            audio[: t * config["hop_size"]],
            config["sampling_rate"],
            "PCM_16",
        )

def main():
    out_dir = "./prediction/melgan.v1/"
    # check directory existence
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # load config
    with open("./vocoder/melgan/conf/melgan.v1.yaml") as f:
        config = yaml.load(f, Loader=yaml.Loader)
        
    mel = preprocess("./datasets/jvs_datasets/jvs009/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav")
    # mel = preprocess("/workspace/StarGAN-VC2-tf/datasets/vcc2018_datasets/vcc2018_training/VCC2TF1/10001.wav")

    t = mel.shape[0]
    t = t-t%4
    mel = mel.T[np.newaxis, :, :t, np.newaxis]

    converter = models.Generator()
    converter(tf.random.uniform((1, 80, 64, 2)))
    converter.load_weights("/workspace/MaskCycleGAN-VC-tf/saved_models/maskcyclegan_vc2/20221205-045950/60000/X2Y.h5")
    mel = converter(np.concatenate([mel, np.ones(mel.shape)], axis = -1), training=False)

    mel = np.squeeze(mel).T[np.newaxis, :, :]
    
    # define model and load checkpoint
    melgan = TFMelGANGenerator(config=MelGANGeneratorConfig(**config["melgan_generator_params"]), name="melgan_generator",)
    melgan._build()
    melgan.load_weights("./vocoder/melgan/checkpoints/generator-1670000.h5")

    # melgan inference.
    generated_audios = melgan(mel)
    # convert to numpy.
    generated_audios = generated_audios.numpy()  # [B, T]

    # save to outdir
    for i, audio in enumerate(generated_audios):
        sf.write(
            os.path.join(out_dir, f"jvs002.wav"),
            audio[: t * config["hop_size"]],
            config["sampling_rate"],
            "PCM_16",
        )


if __name__ == "__main__":
    # main_multiband_hf()
    main_multiband()
    # main()
