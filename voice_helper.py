import numpy as np
import soundfile as sf
import librosa

from setup_args import Args
ARGS = Args()

MEL_BASIS = librosa.filters.mel(
    sr=ARGS.sampling_rate,
    n_fft=ARGS.fft_size,
    n_mels=ARGS.mel_size,
    fmin=ARGS.fmin,
    fmax=ARGS.fmax,
)

def get_mels(file_path):
    x, sr = sf.read(file_path)
    x = x.astype(np.float32)
    
    # check sample rate
    if sr != ARGS.sampling_rate:
        x = librosa.resample(x, orig_sr = sr, target_sr = ARGS.sampling_rate)
        
    x, _ = librosa.effects.trim(
                x,
                top_db=ARGS.trim_threshold_in_db,
                frame_length=ARGS.trim_frame_size,
                hop_length=ARGS.trim_hop_size,
            )
    
    # get spectrogram
    D = librosa.stft(
        x,
        n_fft=ARGS.fft_size,
        hop_length=ARGS.hop_size,
        win_length=ARGS.win_length,
        window=ARGS.window,
        pad_mode="reflect",
    )
    S, _ = librosa.magphase(D)  # (#bins, #frames)

    mel = np.log10(np.maximum(np.dot(MEL_BASIS, S), 1e-10)).T  # (#frames, #bins)
    
    return mel
