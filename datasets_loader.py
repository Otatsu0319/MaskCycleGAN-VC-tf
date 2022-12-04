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
            self.train_dataset_path = os.path.join("gs://stargan-vc2-data", "train")
            self.test_dataset_path = os.path.join("gs://stargan-vc2-data", "test")
        else:
            self.train_dataset_path = os.path.join(".", "datasets", "tf_datasets", "train")
            self.test_dataset_path = os.path.join(".", "datasets", "tf_datasets", "test")

        self.train_shard_num = self.args.num_workers*4
        self.test_shard_num = self.args.num_workers*2

        shard_pattern = "shard_{}.records"
        self.shard_read_pattern = shard_pattern.format("*")
        self.shard_write_pattern = shard_pattern.format("{:08d}")


        self.datasets_dtype = {
            'mel_x': {"numpy_dtype": np.float32, "tensor_dtype": tf.float32},
            'mel_y': {"numpy_dtype": np.float32, "tensor_dtype": tf.float32},
        }

        if self.args.remake_datasets or (not self.args.use_tpu and (not os.path.isdir(self.train_dataset_path) or not os.path.isdir(self.test_dataset_path))):
            self.make_datasets()

        train_dataset_pattern_path = os.path.join(self.train_dataset_path, self.shard_read_pattern)
        test_dataset_pattern_path = os.path.join(self.test_dataset_path, self.shard_read_pattern)

        self.train_shard_files = tf.io.matching_files(train_dataset_pattern_path)
        self.test_shard_files = tf.io.matching_files(test_dataset_pattern_path)
    
        self.rand_generator = rand_generator
        if self.args.mask_mode != "FIF":
            raise ValueError("Mask modes other than FIF are not implemented. They have been reported to degrade accuracy.")

        self.mask_region = tf.zeros([self.args.mel_size, self.args.mask_size,1])

    def to_example(self, data):
        mels_x_ndarray, mels_y_ndarray = data
        feature = {
            'mel_x': _bytes_feature(tf.io.serialize_tensor(mels_x_ndarray)),
            'mel_y': _bytes_feature(tf.io.serialize_tensor(mels_y_ndarray)),
        }
        return tf.train.Example(features=tf.train.Features(feature=feature))


    def parse_example(self, example_proto):
        feature_description = {
            'mel_x': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'mel_y': tf.io.FixedLenFeature([], tf.string, default_value=''),
        }
        parsed_elem = tf.io.parse_example(example_proto, feature_description)
        for key in feature_description.keys():
            parsed_elem[key] = tf.io.parse_tensor(parsed_elem[key], out_type=self.datasets_dtype[key]["tensor_dtype"])

        return list(parsed_elem.values())

    def make_mask(self, mel_x, mel_y):
        rand = self.rand_generator.uniform([], minval=0, maxval=self.args.mask_size, dtype=tf.int32)
        pad_shape = tf.pad(tf.expand_dims(tf.stack((rand, self.args.dataset_t_length - (self.args.mask_size+rand)), axis = 0), axis=0), tf.constant([[1, 1], [0,0]]))
        mask_x = tf.pad(self.mask_region, pad_shape, constant_values=1)
        rand = self.rand_generator.uniform([], minval=0, maxval=self.args.mask_size, dtype=tf.int32)
        pad_shape = tf.pad(tf.expand_dims(tf.stack((rand, self.args.dataset_t_length - (self.args.mask_size+rand)), axis = 0), axis=0), tf.constant([[1, 1], [0,0]]))
        mask_y = tf.pad(self.mask_region, pad_shape, constant_values=1)
        return mel_x, mask_x, mel_y, mask_y
        
        
    def load_dataset(self, batch_size, mode, allow_tensor_cache = False):
        if mode == "train":
            shards = tf.data.Dataset.from_tensor_slices(self.train_shard_files).cache()
            shards = shards.shuffle(self.train_shard_num, reshuffle_each_iteration=True)
        elif mode == "test":
            shards = tf.data.Dataset.from_tensor_slices(self.test_shard_files).cache()
        else:
            raise ValueError("allowed mode is train/test")

        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.AUTO
        shards = shards.with_options(options)
        
        dataset = shards.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)
        
        if allow_tensor_cache:
            dataset = dataset.cache()
        
        dataset = dataset.map(map_func=self.parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(map_func=self.make_mask, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset


    def make_datasets(self):
        pool = multiprocessing.Pool(processes = self.args.preprocess_thread_num)

        npz_path = './datasets/stats'
        os.makedirs(npz_path, exist_ok=True)
        
        def _make_ndarray(data_path_list, scaler):
            mels_ndarray = np.empty((0, self.args.mel_size, self.args.dataset_t_length, 1), self.datasets_dtype["mel_x"]["numpy_dtype"])
        
            perm = np.random.permutation(len(data_path_list))
            data_path_list = data_path_list[perm]
            data_path_list = data_path_list[:int(len(data_path_list) * self.args.preset_datafile_ratio)]

            mel_list = []
            for mel in pool.imap_unordered(get_mels, tqdm.tqdm(data_path_list, total=len(data_path_list), desc="calc mel"), chunksize=20,):
                mel_list.append(mel)
            
            mel_concatenated = np.concatenate(mel_list, axis=0)
            scaler.fit(mel_concatenated) 
            
            for mel in tqdm.tqdm(mel_list, total=len(mel_list), desc="make ndarray"):
                mel_norm = scaler_x.transform(mel)
                mel_norm = mel_norm.astype(self.datasets_dtype["mel_x"]["numpy_dtype"])

                hop_list = list(range(0, len(mel_norm)-self.args.dataset_t_length, self.args.dataset_hop_size))
                hop_list.append(len(mel_norm)-self.args.dataset_t_length)

                for frame in tqdm.tqdm(hop_list, leave=False):
                    mels_frame = np.copy(mel_norm[frame:frame+self.args.dataset_t_length, :]).T
                    mels_ndarray = np.append(mels_ndarray, mels_frame[np.newaxis, :, :, np.newaxis], axis = 0)
            
            perm = np.random.permutation(len(mels_ndarray))
            mels_ndarray = mels_ndarray[perm]
        
            return mels_ndarray, scaler
        
        
        print("make x...")
        data_path_list = np.array(glob.glob(os.path.join(".", "datasets", "jvs_datasets", self.args.using_jvs_id_x, "parallel100", "*", "*.wav")))
        scaler_x = StandardScaler()
        scaler_x.n_features_in_ = self.args.mel_size
        
        mels_x_ndarray, scaler_x = _make_ndarray(data_path_list, scaler_x)
        np.savez(npz_path + f"/{self.args.using_jvs_id_x}_stats", mean=scaler_x.mean_, scale=scaler_x.scale_)
        
        print("make y...")
        data_path_list = np.array(glob.glob(os.path.join(".", "datasets", "jvs_datasets", self.args.using_jvs_id_y, "parallel100", "*", "*.wav")))
        scaler_y = StandardScaler()
        scaler_y.n_features_in_ = self.args.mel_size
        
        mels_y_ndarray, scaler_x = _make_ndarray(data_path_list, scaler_y)
        np.savez(npz_path + f"/{self.args.using_jvs_id_y}_stats", mean=scaler_y.mean_, scale=scaler_y.scale_)

        max_size = min([len(mels_x_ndarray), len(mels_y_ndarray)])


        train_size = int(max_size*self.args.train_data_ratio)
        train_dataset = tf.data.Dataset.from_tensor_slices((mels_x_ndarray[:train_size], mels_y_ndarray[:train_size]))
        test_dataset = tf.data.Dataset.from_tensor_slices((mels_x_ndarray[train_size:max_size], mels_y_ndarray[train_size:max_size]))

        ds_size_path = os.path.join(".", "datasets", "dataset_size.pkl")
        with open(ds_size_path, 'wb') as p:
            pickle.dump([train_size, max_size-train_size], p)

        os.makedirs(self.train_dataset_path, exist_ok=True)
        os.makedirs(self.test_dataset_path, exist_ok=True)

        for i in tqdm.tqdm(range(self.train_shard_num)):
            tfrecords_shard_path = os.path.join(self.train_dataset_path, self.shard_write_pattern.format(i))
            shard_data = train_dataset.shard(self.train_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        for i in tqdm.tqdm(range(self.test_shard_num)):
            tfrecords_shard_path = os.path.join(self.test_dataset_path, self.shard_write_pattern.format(i))
            test_dataset.shard(self.test_shard_num, i)
            with tf.io.TFRecordWriter(tfrecords_shard_path) as writer:
                for data in shard_data:
                    tf_example = self.to_example(data)
                    writer.write(tf_example.SerializeToString())

        del train_dataset
        del test_dataset
        gc.collect()


if __name__ == "__main__":
    import setup_args
    import pprint
    args = setup_args.Args()
    np.set_printoptions(threshold=np.inf)
    rand_generator = tf.random.Generator.from_non_deterministic_state()
    
    rand = rand_generator.uniform([], minval=0, maxval=args.mask_size, dtype=tf.int32)

    data = DatasetLoader(args, rand_generator)
    # data.make_datasets()
    train_data = data.load_dataset(128, "train")

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

    test_data = data.load_dataset(128, "test")

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