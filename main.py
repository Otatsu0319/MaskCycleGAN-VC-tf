#!/usr/bin/python3

import tensorflow as tf
import setup_args
from maskcyclegan_vc import MaskCycleGAN_VC

if __name__ == "__main__":
    args = setup_args.Args()

    if args.set_seed:
        tf.config.threading.set_inter_op_parallelism_threads(1)

    if args.use_tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu='')
        tf.config.experimental_connect_to_cluster(resolver)
        # This is the TPU initialization code that has to be at the beginning.
        tf.tpu.experimental.initialize_tpu_system(resolver)
        # print("All devices: ", tf.config.list_logical_devices('TPU'))
        tf.config.set_soft_device_placement(True)
        
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if len(gpus) > 0:
        [tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]

        # https://github.com/tensorflow/tensorflow/issues/56661#issuecomment-1213286290
        tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=11200)]) 
    
    if args.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16" if not args.use_tpu else "mixed_bfloat16")



    if args.use_tpu:
        maskcyclegan_vc = MaskCycleGAN_VC(args, resolver)
    else:
        maskcyclegan_vc = MaskCycleGAN_VC(args)

    maskcyclegan_vc.train()