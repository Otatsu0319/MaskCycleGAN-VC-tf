import glob
import os
import re
import tensorflow as tf
import tqdm
import numpy as np
import random
import pickle
import gc
import multiprocessing
from functools import partial
from sklearn.preprocessing import StandardScaler

from voice_helper import get_mels

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if isinstance(value, type(tf.constant(0))):
    value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class DatasetLoader():
    def __init__(self, args, rand_generator):
        self.args = args
        if self.args.set_seed:
            random.seed(self.args.seed)

        if self.args.use_tpu:
            raise ValueError("This dataset is not TPU ready")
        else:
            self.x_npy_path = os.path.join(".", "datasets", "mel_datasets", self.args.using_jvs_id_x)
            self.y_npy_path = os.path.join(".", "datasets", "mel_datasets", self.args.using_jvs_id_y)

        self.datasets_dtype = {
            'mel_x': {"numpy_dtype": np.float32, "tensor_dtype": tf.float32},
            'mel_y': {"numpy_dtype": np.float32, "tensor_dtype": tf.float32},
        }

        if self.args.remake_datasets or (not self.args.use_tpu and (not os.path.isdir(self.x_npy_path) or not os.path.isdir(self.y_npy_path))):
            self.make_datasets()

        x_npy_list = glob.glob(os.path.join(self.x_npy_path, "*.npy"))
        y_npy_list = glob.glob(os.path.join(self.y_npy_path, "*.npy"))
        max_size = min([len(x_npy_list), len(y_npy_list)])

        self.x_npy_list = x_npy_list[:int(max_size * self.args.preset_datafile_ratio)]
        self.y_npy_list = y_npy_list[:int(max_size * self.args.preset_datafile_ratio)]
        
        self.train_size = int(len(self.x_npy_list)*self.args.train_data_ratio)

        self.rand_generator = rand_generator
        if self.args.mask_mode != "FIF":
            raise ValueError("Mask modes other than FIF are not implemented. They have been reported to degrade accuracy.")

        self.mask_region = tf.zeros([self.args.mel_size, self.args.mask_size,1])

    @tf.function
    def load_npy(self, path):
        mel = tf.numpy_function(np.load, [path], tf.float32)
        return mel
    
    @tf.function
    def rand_crop(self, mel_x, mel_y):
        seed = self.rand_generator.uniform_full_int([2], dtype=tf.int32)
        mel_x = tf.image.stateless_random_crop(mel_x, [self.args.mel_size, self.args.dataset_t_length, 1], seed)
        
        seed = self.rand_generator.uniform_full_int([2], dtype=tf.int32)
        mel_y = tf.image.stateless_random_crop(mel_y, [self.args.mel_size, self.args.dataset_t_length, 1], seed)
        
        return mel_x, mel_y

    @tf.function
    def make_mask(self, mel_x, mel_y):
        rand = self.rand_generator.uniform([], minval=0, maxval=self.args.mask_size, dtype=tf.int32)
        pad_shape = tf.pad(tf.expand_dims(tf.stack((rand, self.args.dataset_t_length - (self.args.mask_size+rand)), axis = 0), axis=0), tf.constant([[1, 1], [0,0]]))
        mask_x = tf.pad(self.mask_region, pad_shape, constant_values=1)
        rand = self.rand_generator.uniform([], minval=0, maxval=self.args.mask_size, dtype=tf.int32)
        pad_shape = tf.pad(tf.expand_dims(tf.stack((rand, self.args.dataset_t_length - (self.args.mask_size+rand)), axis = 0), axis=0), tf.constant([[1, 1], [0,0]]))
        mask_y = tf.pad(self.mask_region, pad_shape, constant_values=1)
        return mel_x, mask_x, mel_y, mask_y

        
    def load_dataset(self, batch_size, mode, allow_tensor_cache = True):
        if mode == "train":
            x_dataset = tf.data.Dataset.from_tensor_slices((self.x_npy_list[:self.train_size]))
            y_dataset = tf.data.Dataset.from_tensor_slices((self.y_npy_list[:self.train_size]))
        elif mode == "test":
            x_dataset = tf.data.Dataset.from_tensor_slices((self.x_npy_list[self.train_size:]))
            y_dataset = tf.data.Dataset.from_tensor_slices((self.y_npy_list[self.train_size:]))
        else:
            raise ValueError("allowed mode is train/test")

        x_dataset = x_dataset.map(map_func=self.load_npy, num_parallel_calls=tf.data.AUTOTUNE)
        y_dataset = y_dataset.map(map_func=self.load_npy, num_parallel_calls=tf.data.AUTOTUNE)
         
        if allow_tensor_cache:
            x_dataset = x_dataset.cache()
            y_dataset = y_dataset.cache()
                   
        if mode == "train":
            x_dataset = x_dataset.shuffle(self.train_size)
            y_dataset = y_dataset.shuffle(self.train_size)
            
        dataset = tf.data.Dataset.zip((x_dataset, y_dataset))
        
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)

        dataset = dataset.map(map_func=self.rand_crop, num_parallel_calls=tf.data.AUTOTUNE)
        
        dataset = dataset.map(map_func=self.make_mask, num_parallel_calls=tf.data.AUTOTUNE)

        if mode == "train":
            dataset = dataset.repeat(self.args.repeat_num)
        
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


    def make_datasets(self):
        pool = multiprocessing.Pool(processes = self.args.preprocess_thread_num)

        stats_path = './datasets/stats'
        os.makedirs(stats_path, exist_ok=True)
        
        def _make_ndarray(using_jvs_id, scaler):
            data_path_list = glob.glob(os.path.join(".", "datasets", "jvs_datasets", using_jvs_id, "parallel100", "*", "*.wav"))
     
            np_save_path = os.path.join('.', "datasets", "mel_datasets", using_jvs_id)
            os.makedirs(np_save_path, exist_ok=True)

            mel_list = []
            for mel in pool.imap_unordered(get_mels, tqdm.tqdm(data_path_list, total=len(data_path_list), desc="calc mel"), chunksize=20,):
                mel_list.append(mel)
            
            mel_concatenated = np.concatenate(mel_list, axis=0)
            scaler.fit(mel_concatenated) 
            
            for i, mel in enumerate(tqdm.tqdm(mel_list, total=len(mel_list), desc="make ndarray")):
                save_path = os.path.join(np_save_path, f"mel_{i:03}")
                mel_norm = scaler_x.transform(mel)
                mel_norm = mel_norm.astype(self.datasets_dtype["mel_x"]["numpy_dtype"])

                np.save(save_path, mel_norm.T[:, :, np.newaxis])
        
            return scaler
        
        
        print("make np mel x...")
        scaler_x = StandardScaler()
        scaler_x.n_features_in_ = self.args.mel_size
        
        scaler_x = _make_ndarray(self.args.using_jvs_id_x, scaler_x)
        np.savez(stats_path + f"/{self.args.using_jvs_id_x}_stats", mean=scaler_x.mean_, scale=scaler_x.scale_)
        
        print("make np mel y...")
        scaler_y = StandardScaler()
        scaler_y.n_features_in_ = self.args.mel_size
        
        scaler_y = _make_ndarray(self.args.using_jvs_id_y, scaler_y)
        np.savez(stats_path + f"/{self.args.using_jvs_id_y}_stats", mean=scaler_y.mean_, scale=scaler_y.scale_)


if __name__ == "__main__":
    import setup_args
    import pprint
    args = setup_args.Args()
    np.set_printoptions(threshold=np.inf)
    rand_generator = tf.random.Generator.from_non_deterministic_state()
    
    rand = rand_generator.uniform([], minval=0, maxval=args.mask_size, dtype=tf.int32)

    data = DatasetLoader(args, rand_generator)
    # data.make_datasets()
    train_data = data.load_dataset(4, "train")

    print(train_data)
    for i, datas in enumerate(train_data):
        print(i)
        mel_x, mask_x, mel_y, mask_y = datas
        print(mel_x.shape)
        print(mask_x.shape)
        print(mel_y.shape)
        print(mask_y.shape)
        if i==3:
            break

    test_data = data.load_dataset(4, "test")

    print(test_data)
    for i, datas in enumerate(test_data):
        print(i)
        mel_x, mask_x, mel_y, mask_y = datas
        print(mel_x.shape)
        print(mask_x.shape)
        print(mel_y.shape)
        print(mask_y.shape)
        # if i==3:
        #     break