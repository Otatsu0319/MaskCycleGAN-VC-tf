import tensorflow as tf
import os
import pickle
import numpy as np

import pyworld as pw
import pysptk as sptk
import soundfile as sf

import voice_helper
import models
from setup_args import Args
from stargan_vc2 import StarGAN_VC2
ARGS = Args()

if __name__ == "__main__":        
    onehot_dict_path = os.path.join(".", "datasets", "onehot_dict.pkl")
    with open(onehot_dict_path, 'rb') as p:
        onehot_vectors = pickle.load(p)  

    dist_dict_path = os.path.join(".", "datasets", "dist_dict.pkl")
    with open(dist_dict_path, 'rb') as p:
        distribution = pickle.load(p)  

    args = Args()

    generator = tf.saved_model.load("./saved_models/stargan_vc2/20220906-084415/00050")
    # load_generator = tf.keras.models.load_model("./saved_models/stargan_vc2/20220906-084415/00050", compile = False)
    # generator = models.Generator(args.code_size)
    # generator((tf.random.uniform((1, 35, 444, 1)), tf.random.uniform((1, 8)))) # build
    # generator.set_weights(load_generator.get_weights()) 

    source_id = "jvs009"
    target_id = "jvs002"
    source_path = "./datasets/jvs_datasets/" + source_id + "/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"
    target_path = "./datasets/jvs_datasets/" + target_id + "/parallel100/wav24kHz16bit/VOICEACTRESS100_001.wav"

    s_f0, s_mceps, s_ap, s_sr = voice_helper.get_conversion_data(source_path)
    t = s_mceps
    print("orig source", np.min(t), np.max(t), np.var(t), np.std(t))

    s_mceps = s_mceps.astype(np.float32)
    
    mceps_mean, mceps_std = distribution[source_id]
    s_mceps = (s_mceps - mceps_mean) / mceps_std

    t = s_mceps
    print("std source", np.min(t), np.max(t), np.var(t), np.std(t))

    pad_size = (4 - len(s_mceps) % 4)
    s_mceps = np.pad(s_mceps, [(0, pad_size),(0,0)], 'constant')

    s_mceps_ndarray = s_mceps.T[np.newaxis, :, :, np.newaxis]
    target_ndarray = onehot_vectors[target_id][np.newaxis, :]

    t_mceps_tensor = generator((s_mceps_ndarray, target_ndarray))
    t_mceps_ndarray = t_mceps_tensor.numpy()
    t_mceps = np.squeeze(t_mceps_ndarray).T.copy(order='C')

    t = t_mceps
    print("gen tgt", np.min(t), np.max(t), np.var(t), np.std(t))

    mceps_mean, mceps_std = distribution[target_id]
    t_mceps = (t_mceps * mceps_std) + mceps_mean

    t = t_mceps
    print("gunstd tgt", np.min(t), np.max(t), np.var(t), np.std(t))

    voice = voice_helper.synthesize_convert_voice(s_f0, t_mceps[:-pad_size], s_ap, s_sr, target_path, write_path = "./conversion_data/9to2_convert.wav")

    print("9to2 done!")

    source_ndarray = onehot_vectors[source_id][np.newaxis, :]

    t_mceps_tensor = generator((s_mceps_ndarray, source_ndarray))
    t_mceps_ndarray = t_mceps_tensor.numpy()
    t_mceps = np.squeeze(t_mceps_ndarray).T.copy(order='C')

    mceps_mean, mceps_std = distribution[source_id]
    t_mceps = (t_mceps * mceps_std) + mceps_mean
    
    voice = voice_helper.synthesize_convert_voice(s_f0, t_mceps[:-pad_size], s_ap, s_sr, source_path, write_path = "./conversion_data/9to9_convert.wav")

    print("9to9 done!")