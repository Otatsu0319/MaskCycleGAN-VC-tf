from itertools import cycle
import tensorflow as tf
import numpy as np
import os
import io
import json
import random
import pickle
from tqdm import tqdm

from setup_args import Args
from datasets_loader import DatasetLoader
import models

class MaskCycleGAN_VC():
    def __init__(self, args: Args, resolver = None):
        self._load_args(args)
        
        if args.set_seed:
            seed = args.seed

            os.environ["PYTONHASHSEED"] = '0'
            tf.random.set_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            os.environ["TF_DETERMINISTIC_OPS"] = '0'
            os.environ["TF_CUDNN_DETERMINISTIC"] = '0'

        if args.use_tpu:
            self.distribute_strategy = tf.distribute.TPUStrategy(resolver)
            root_dir = "gs://stargan-vc2-data"
        
        else:
            self.distribute_strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0"]) # only 1 gpu using. if you using 2 gpu : devices=["/gpu:0", "/gpu:1"]
            root_dir = os.path.dirname(__file__)
        
        self.checkpoint_dir = os.path.join(root_dir, "training_checkpoints", self.model_name, self.datetime, "ckpt")
        self.log_dir = os.path.join(root_dir, self.tensorboard_log_dir, self.model_name, self.datetime)
        self.savedmodel_dir = os.path.join(root_dir, "saved_weights", self.model_name, self.datetime)

        self._compile_learn_function()
        
        with self.distribute_strategy.scope():
            if args.set_seed:
                self.rand_generator = tf.random.Generator.from_seed(seed)
            else:
                self.rand_generator = tf.random.Generator.from_non_deterministic_state()
                
                
            # model initialize
            self.generator_X2Y = models.Generator()
            self.generator_Y2X = models.Generator()
            # for one step adversarial loss
            self.discriminator_X = models.Discriminator()
            self.discriminator_Y = models.Discriminator()
            # for two step adversarial loss
            self.discriminator_X2 = models.Discriminator()
            self.discriminator_Y2 = models.Discriminator()

        
            # optimizer initialize
            self.g_X2Y_optimizer = tf.keras.optimizers.Adam(learning_rate=args.g_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            self.g_Y2X_optimizer = tf.keras.optimizers.Adam(learning_rate=args.g_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            self.d_X_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            self.d_Y_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            self.d_X2_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            self.d_Y2_optimizer = tf.keras.optimizers.Adam(learning_rate=args.d_learn_rate, beta_1 = args.beta_1, beta_2 = args.beta_2)
            if args.mixed_precision and not args.use_tpu:
                self.g_X2Y_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.g_X2Y_optimizer)
                self.g_Y2X_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.g_Y2X_optimizer)
                self.d_X_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_X_optimizer)
                self.d_Y_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_Y_optimizer)
                self.d_X2_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_X2_optimizer)
                self.d_Y2_optimizer = tf.keras.mixed_precision.LossScaleOptimizer(self.d_Y2_optimizer)

            # loss objects
            self.bc = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
            self.mae = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

            # metrics objects
            self._metrics_init()

            # checkpoint init
            self.checkpoint = tf.train.Checkpoint(g_x2y = self.generator_X2Y, g_x2y_opt = self.g_X2Y_optimizer,
                                                  g_y2x = self.generator_Y2X, g_y2x_opt = self.g_Y2X_optimizer,
                                                  d_x = self.discriminator_X, d_x_opt = self.d_X_optimizer,
                                                  d_y = self.discriminator_Y, d_y_opt = self.d_Y_optimizer,
                                                  d_x2 = self.discriminator_X2, d_x2_opt = self.d_X2_optimizer, 
                                                  d_y2 = self.discriminator_Y2, d_y2_opt = self.d_Y2_optimizer, )
            self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

            if self.restore_bool:
                self.checkpoint.restore(self.checkpoint_manager.latest_checkpoint)
                print("restored model...")

        self.datasets = DatasetLoader(args, self.rand_generator)
        
        self.train_dataset = self.distribute_strategy.experimental_distribute_dataset(self.datasets.load_dataset(args.train_batch_size, mode="train"))
        self.test_dataset = self.distribute_strategy.experimental_distribute_dataset(self.datasets.load_dataset(args.test_batch_size, mode="test"))

        self.per_replica_batch_size = args.train_batch_size // self.distribute_strategy.num_replicas_in_sync


    #####################
    ## small task func ##
    #####################
    def _load_args(self, args:Args):
        self.args_dict = args.__dict__
        
        self.model_name = args.model_name
        self.datetime = args.datetime
        self.tensorboard_log_dir = args.tensorboard_log_dir
        self.restore_bool = args.restore_bool
        self.mixed_precision = args.mixed_precision
        self.use_tpu = args.use_tpu
        self.lambda_cyc = args.lambda_cyc
        self.lambda_id = args.lambda_id
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.dataset_t_length = args.dataset_t_length
        self.mel_size = args.mel_size
        self.repeat_num = args.repeat_num
        self.iterations = args.iterations
        self.start_iteration = args.start_iteration
        self.save_profile = args.save_profile
        self.id_rate = args.id_rate
        
        self.sample_interval = args.sample_interval
        self.logging_interval = args.logging_interval
        self.checkpoint_interval = args.checkpoint_interval

        self.mask_mode = args.mask_mode
        self.mask_size = args.mask_size
    
    def _metrics_init(self):
        self.train_loss_X2Y = tf.keras.metrics.Mean("train_loss_X2Y", dtype=tf.float32)
        self.train_loss_Y2X = tf.keras.metrics.Mean("train_loss_Y2X", dtype=tf.float32)
        self.train_loss_X = tf.keras.metrics.Mean("train_loss_X", dtype=tf.float32)
        self.train_loss_Y = tf.keras.metrics.Mean("train_loss_Y", dtype=tf.float32)
        self.train_loss_X2 = tf.keras.metrics.Mean("train_loss_X2", dtype=tf.float32)
        self.train_loss_Y2 = tf.keras.metrics.Mean("train_loss_Y2", dtype=tf.float32)
        
        self.test_loss_X2Y = tf.keras.metrics.Mean("train_loss_X2Y", dtype=tf.float32)
        self.test_loss_Y2X = tf.keras.metrics.Mean("train_loss_Y2X", dtype=tf.float32)
        self.test_loss_X = tf.keras.metrics.Mean("train_loss_X", dtype=tf.float32)
        self.test_loss_Y = tf.keras.metrics.Mean("train_loss_Y", dtype=tf.float32)
        self.test_loss_X2 = tf.keras.metrics.Mean("train_loss_X2", dtype=tf.float32)
        self.test_loss_Y2 = tf.keras.metrics.Mean("train_loss_Y2", dtype=tf.float32)

    def _summary_write(self, iteration, melspecs = None, d_outputs = None):
        train_loss_X2Y = self.train_loss_X2Y.result()
        train_loss_Y2X = self.train_loss_Y2X.result()
        train_loss_X = self.train_loss_X.result()
        train_loss_Y = self.train_loss_Y.result()
        train_loss_X2 = self.train_loss_X2.result()
        train_loss_Y2 = self.train_loss_Y2.result()
        
        test_loss_X2Y = self.test_loss_X2Y.result()
        test_loss_Y2X = self.test_loss_Y2X.result()
        test_loss_X = self.test_loss_X.result()
        test_loss_Y = self.test_loss_Y.result()
        test_loss_X2 = self.test_loss_X2.result()
        test_loss_Y2 = self.test_loss_Y2.result()

        # origin_melspecs, generate_melspecs, cycle_melspecs, identity_melspecs = melspecs
        # y_real, y_fake = d_output

        # TensorBoardにloss類と元/生成melspecを保存
        with self.summary_writer.as_default():
            tf.summary.scalar("train/train_loss_X2Y", train_loss_X2Y, iteration)
            tf.summary.scalar("train/train_loss_Y2X", train_loss_Y2X, iteration)
            tf.summary.scalar("train/train_loss_X", train_loss_X, iteration)
            tf.summary.scalar("train/train_loss_Y", train_loss_Y, iteration)
            tf.summary.scalar("train/train_loss_X2", train_loss_X2, iteration)
            tf.summary.scalar("train/train_loss_Y2", train_loss_Y2, iteration)
            
            tf.summary.scalar("test/test_loss_X2Y", test_loss_X2Y, iteration)
            tf.summary.scalar("test/test_loss_Y2X", test_loss_Y2X, iteration)
            tf.summary.scalar("test/test_loss_X", test_loss_X, iteration)
            tf.summary.scalar("test/test_loss_Y", test_loss_Y, iteration)
            tf.summary.scalar("test/test_loss_X2", test_loss_X2, iteration)
            tf.summary.scalar("test/test_loss_Y2", test_loss_Y2, iteration)

            if melspecs:
                real_X, real_Y, mask_X, mask_Y, fake_X, fake_Y, cycle_X, cycle_Y, identity_X, identity_Y = melspecs
                if iteration == 1:
                    tf.summary.image("origin_melspecs_X", real_X, iteration)
                    tf.summary.image("origin_melspecs_Y", real_X, iteration)
                tf.summary.image("generate_melspecs_X", fake_X, iteration)
                tf.summary.image("cycle_melspecs_X", cycle_X, iteration)
                tf.summary.image("identity_melspecs_X", identity_X, iteration)
                tf.summary.text("test_summary/gen_melspec_min_X", tf.strings.as_string(tf.math.reduce_min(real_X[0])), iteration)
                tf.summary.text("test_summary/gen_melspec_max_X", tf.strings.as_string(tf.math.reduce_max(real_X[0])), iteration)
                tf.summary.image("generate_melspecs_Y", fake_Y, iteration)
                tf.summary.image("cycle_melspecs_Y", cycle_Y, iteration)
                tf.summary.image("identity_melspecs_Y", identity_Y, iteration)
                tf.summary.text("test_summary/gen_melspec_min_Y", tf.strings.as_string(tf.math.reduce_min(real_Y[0])), iteration)
                tf.summary.text("test_summary/gen_melspec_max_Y", tf.strings.as_string(tf.math.reduce_max(real_Y[0])), iteration)

            if d_outputs:
                y_real, y_fake = d_outputs
                tf.summary.text("test_summary/y_real", tf.strings.as_string(y_real), iteration)
                tf.summary.text("test_summary/y_fake", tf.strings.as_string(y_fake), iteration)

    def _reset_state(self):
        self.train_loss_X2Y.reset_state()
        self.train_loss_Y2X.reset_state()
        self.train_loss_X.reset_state()
        self.train_loss_Y.reset_state()
        self.train_loss_X2.reset_state()
        self.train_loss_Y2.reset_state()
        
        self.test_loss_X2Y.reset_state()
        self.test_loss_Y2X.reset_state()
        self.test_loss_X.reset_state()
        self.test_loss_Y.reset_state()
        self.test_loss_X2.reset_state()
        self.test_loss_Y2.reset_state()

    def _compile_learn_function(self):
        if self.mixed_precision and not self.use_tpu:
            self.train_step_func = tf.function(self.train_step_mp, jit_compile=True)
            self.test_step_func = tf.function(self.test_step, jit_compile=True)
        else:
            self.train_step_func = tf.function(self.train_step)
            self.test_step_func = tf.function(self.test_step)

    ###############
    ## loss func ##
    ###############
    def discriminator_loss(self, d_real, d_fake, batch_size):
        # 1 = real
        # this loss is very easy, so if learning baranss mismatched then don't using this
        # loss_real = self.bc(tf.ones_like(d_real), d_real)
        # loss_fake = self.bc(tf.zeros_like(d_fake), d_fake)

        # this loss is very difficult
        # LSGAN
        loss_real = tf.reduce_mean((1.0 - d_real) ** 2, axis=1)
        loss_fake = tf.reduce_mean((d_fake) ** 2, axis=1)

        per_example_loss_real = tf.nn.compute_average_loss(loss_real, global_batch_size=batch_size)
        per_example_loss_fake = tf.nn.compute_average_loss(loss_fake, global_batch_size=batch_size)
        return (per_example_loss_real + per_example_loss_fake) / 2.0

    def generator_loss(self, real_m, cycle_m, identity_m, d_fake, d_fake_2, batch_size):
        per_loss_cyc = self.mae(real_m, cycle_m)
        per_loss_id = self.mae(real_m, identity_m)
        
        # per_loss_adv = self.bc(tf.ones_like(d_fake), d_fake)
        # per_loss_adv2 = self.bc(tf.ones_like(d_fake_2), d_fake_2)
        
        # LSGAN
        per_loss_adv = tf.reduce_mean((1.0 - d_fake) ** 2, axis=1)
        per_loss_adv2 = tf.reduce_mean((1.0 - d_fake_2) ** 2, axis=1)

        per_example_loss_adv = tf.nn.compute_average_loss(per_loss_adv, global_batch_size=batch_size)
        per_example_loss_adv2 = tf.nn.compute_average_loss(per_loss_adv2, global_batch_size=batch_size)
        total_loss_cyc = tf.nn.compute_average_loss(per_loss_cyc, global_batch_size=batch_size * self.dataset_t_length * self.mel_size)
        total_loss_id = tf.nn.compute_average_loss(per_loss_id, global_batch_size=batch_size * self.dataset_t_length * self.mel_size)
        return per_example_loss_adv, per_example_loss_adv2, total_loss_cyc, self.lambda_id * total_loss_id


    #####################
    ## train/test func ##
    #####################
    
    def train_step_mp(self, dataset_inputs):

        def step_fn(inputs):
            real_X, mask_X, real_Y, mask_Y = inputs
            real_X_masked = tf.math.multiply(real_X, mask_X)
            real_Y_masked = tf.math.multiply(real_Y, mask_Y)
            
            with tf.GradientTape(persistent=True) as tape:
                fake_Y = self.generator_X2Y(tf.concat((real_X_masked, mask_X), -1))
                cycle_X = self.generator_Y2X(tf.concat((fake_Y, tf.ones_like(fake_Y)), -1))

                fake_X = self.generator_Y2X(tf.concat((real_Y_masked, mask_Y), -1))
                cycle_Y = self.generator_X2Y(tf.concat((fake_X, tf.ones_like(fake_X)), -1))

                identity_X = self.generator_Y2X(tf.concat((real_X, tf.ones_like(real_X)), -1))
                identity_Y = self.generator_X2Y(tf.concat((real_Y, tf.ones_like(real_Y)), -1))

                d_real_X = self.discriminator_X(real_X)
                d_real_Y = self.discriminator_Y(real_Y)
                d_fake_X = self.discriminator_X(fake_X)
                d_fake_Y = self.discriminator_Y(fake_Y)

                d_real_X2 = self.discriminator_X2(real_X)
                d_real_Y2 = self.discriminator_Y2(real_Y)
                d_fake_X2 = self.discriminator_X2(cycle_X)
                d_fake_Y2 = self.discriminator_Y2(cycle_Y)


                loss_adv_X2Y, loss_adv2_X2Y, loss_cyc_X2Y, loss_id_X2Y= self.generator_loss(real_Y, cycle_Y, identity_Y, d_fake_Y, d_fake_Y2, self.train_batch_size)
                loss_adv_Y2X, loss_adv2_Y2X, loss_cyc_Y2X, loss_id_Y2X= self.generator_loss(real_X, cycle_X, identity_X, d_fake_X, d_fake_X2, self.train_batch_size)
                total_cycle_loss = self.lambda_cyc * (loss_cyc_X2Y + loss_cyc_Y2X)

                loss_X2Y = loss_adv_X2Y + loss_adv2_X2Y + total_cycle_loss + loss_id_X2Y
                loss_Y2X = loss_adv_Y2X + loss_adv2_Y2X + total_cycle_loss + loss_id_Y2X
                
                loss_X = self.discriminator_loss(d_real_X, d_fake_X, self.train_batch_size)
                loss_Y = self.discriminator_loss(d_real_Y, d_fake_Y, self.train_batch_size)

                loss_X2 = self.discriminator_loss(d_real_X2, d_fake_X2, self.train_batch_size)
                loss_Y2 = self.discriminator_loss(d_real_Y2, d_fake_Y2, self.train_batch_size)

                scaled_loss_X2Y = self.g_X2Y_optimizer.get_scaled_loss(loss_X2Y)
                scaled_loss_Y2X = self.g_Y2X_optimizer.get_scaled_loss(loss_Y2X)
                scaled_loss_X = self.d_X_optimizer.get_scaled_loss(loss_X)
                scaled_loss_Y = self.d_Y_optimizer.get_scaled_loss(loss_Y)
                scaled_loss_X2 = self.d_X2_optimizer.get_scaled_loss(loss_X2)
                scaled_loss_Y2 = self.d_Y2_optimizer.get_scaled_loss(loss_Y2)

            scaled_gradients_X2Y = tape.gradient(scaled_loss_X2Y, self.generator_X2Y.trainable_variables)
            scaled_gradients_Y2X = tape.gradient(scaled_loss_Y2X, self.generator_Y2X.trainable_variables)
            scaled_gradients_X = tape.gradient(scaled_loss_X, self.discriminator_X.trainable_variables)
            scaled_gradients_Y = tape.gradient(scaled_loss_Y, self.discriminator_Y.trainable_variables)
            scaled_gradients_X2 = tape.gradient(scaled_loss_X2, self.discriminator_X2.trainable_variables)
            scaled_gradients_Y2 = tape.gradient(scaled_loss_Y2, self.discriminator_Y2.trainable_variables)

            gradients_X2Y = self.g_X2Y_optimizer.get_unscaled_gradients(scaled_gradients_X2Y)
            gradients_Y2X = self.g_Y2X_optimizer.get_unscaled_gradients(scaled_gradients_Y2X)
            gradients_X = self.d_X_optimizer.get_unscaled_gradients(scaled_gradients_X)
            gradients_Y = self.d_Y_optimizer.get_unscaled_gradients(scaled_gradients_Y)
            gradients_X2 = self.d_X2_optimizer.get_unscaled_gradients(scaled_gradients_X2)
            gradients_Y2 = self.d_Y2_optimizer.get_unscaled_gradients(scaled_gradients_Y2)

            self.g_X2Y_optimizer.apply_gradients(zip(gradients_X2Y, self.generator_X2Y.trainable_variables))
            self.g_Y2X_optimizer.apply_gradients(zip(gradients_Y2X, self.generator_Y2X.trainable_variables))
            self.d_X_optimizer.apply_gradients(zip(gradients_X, self.discriminator_X.trainable_variables))
            self.d_Y_optimizer.apply_gradients(zip(gradients_Y, self.discriminator_Y.trainable_variables))
            self.d_X2_optimizer.apply_gradients(zip(gradients_X2, self.discriminator_X2.trainable_variables))
            self.d_Y2_optimizer.apply_gradients(zip(gradients_Y2, self.discriminator_Y2.trainable_variables))


            self.train_loss_X2Y.update_state(loss_X2Y * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y2X.update_state(loss_Y2X * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_X.update_state(loss_X * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y.update_state(loss_Y * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_X2.update_state(loss_X2 * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y2.update_state(loss_Y2 * self.distribute_strategy.num_replicas_in_sync)

        self.distribute_strategy.run(step_fn, args=(dataset_inputs,))


    def train_step(self, dataset_inputs):
        def step_fn(inputs):
            real_X, mask_X, real_Y, mask_Y = inputs
            
            with tf.GradientTape(persistent=True) as tape:
                fake_Y = self.generator_X2Y(tf.concat((real_X, mask_X), 1))
                cycle_X = self.generator_Y2X(tf.concat((fake_Y, tf.ones_like(fake_Y)), 1))

                fake_X = self.generator_Y2X(tf.concat((real_Y, mask_Y), 1))
                cycle_Y = self.generator_X2Y(tf.concat((fake_X, tf.ones_like(fake_X)), 1))

                identity_X = self.generator_Y2X(tf.concat((real_X, tf.ones_like(real_X)), 1))
                identity_Y = self.generator_X2Y(tf.concat((real_Y, tf.ones_like(real_Y)), 1))

                d_real_X = self.discriminator_X(real_X)
                d_real_Y = self.discriminator_Y(real_Y)
                d_fake_X = self.discriminator_X(fake_X)
                d_fake_Y = self.discriminator_Y(fake_Y)

                d_real_X2 = self.discriminator_X2(real_X)
                d_real_Y2 = self.discriminator_Y2(real_Y)
                d_fake_X2 = self.discriminator_X2(cycle_X)
                d_fake_Y2 = self.discriminator_Y2(cycle_Y)


                loss_adv_X2Y, loss_adv2_X2Y, loss_cyc_X2Y, loss_id_X2Y= self.generator_loss(real_Y, cycle_Y, identity_Y, d_fake_Y, d_fake_Y2, self.train_batch_size)
                loss_adv_Y2X, loss_adv2_Y2X, loss_cyc_Y2X, loss_id_Y2X= self.generator_loss(real_X, cycle_X, identity_X, d_fake_X, d_fake_X2, self.train_batch_size)
                total_cycle_loss = self.lambda_cyc * (loss_cyc_X2Y + loss_cyc_Y2X)

                loss_X2Y = loss_adv_X2Y + loss_adv2_X2Y + total_cycle_loss + loss_id_X2Y
                loss_Y2X = loss_adv_Y2X + loss_adv2_Y2X + total_cycle_loss + loss_id_Y2X
                
                loss_X = self.discriminator_loss(d_real_X, d_fake_X, self.train_batch_size)
                loss_Y = self.discriminator_loss(d_real_Y, d_fake_Y, self.train_batch_size)

                loss_X2 = self.discriminator_loss(d_real_X2, d_fake_X2, self.train_batch_size)
                loss_Y2 = self.discriminator_loss(d_real_Y2, d_fake_Y2, self.train_batch_size)


            gradients_X2Y = tape.gradient(loss_X2Y, self.generator_X2Y.trainable_variables)
            gradients_Y2X = tape.gradient(loss_Y2X, self.generator_Y2X.trainable_variables)
            gradients_X = tape.gradient(loss_X, self.discriminator_X.trainable_variables)
            gradients_Y = tape.gradient(loss_Y, self.discriminator_Y.trainable_variables)
            gradients_X2 = tape.gradient(loss_X2, self.discriminator_X2.trainable_variables)
            gradients_Y2 = tape.gradient(loss_Y2, self.discriminator_Y2.trainable_variables)

            self.g_X2Y_optimizer.applygradients(zip(gradients_X2Y, self.generator_X2Y.trainable_variables))
            self.g_Y2X_optimizer.applygradients(zip(gradients_Y2X, self.generator_Y2X.trainable_variables))
            self.d_X_optimizer.applygradients(zip(gradients_X, self.discriminator_X.trainable_variables))
            self.d_Y_optimizer.applygradients(zip(gradients_Y, self.discriminator_Y.trainable_variables))
            self.d_X2_optimizer.applygradients(zip(gradients_X2, self.discriminator_X2.trainable_variables))
            self.d_Y2_optimizer.applygradients(zip(gradients_Y2, self.discriminator_Y2.trainable_variables))


            self.train_loss_X2Y.update_state(loss_X2Y * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y2X.update_state(loss_Y2X * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_X.update_state(loss_X * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y.update_state(loss_Y * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_X2.update_state(loss_X2 * self.distribute_strategy.num_replicas_in_sync)
            self.train_loss_Y2.update_state(loss_Y2 * self.distribute_strategy.num_replicas_in_sync)

        self.distribute_strategy.run(step_fn, args=(dataset_inputs,))

        
    def test_step(self, dataset_inputs):
        def step_fn(inputs):
            real_X, mask_X, real_Y, mask_Y = inputs
            real_X_masked = tf.math.multiply(real_X, mask_X)
            real_Y_masked = tf.math.multiply(real_Y, mask_Y)
            
            fake_Y = self.generator_X2Y(tf.concat((real_X_masked, mask_X), -1), training=False)
            cycle_X = self.generator_Y2X(tf.concat((fake_Y, tf.ones_like(fake_Y)), -1), training=False)

            fake_X = self.generator_Y2X(tf.concat((real_Y_masked, mask_Y), -1), training=False)
            cycle_Y = self.generator_X2Y(tf.concat((fake_X, tf.ones_like(fake_X)), -1), training=False)
            
            identity_X = self.generator_Y2X(tf.concat((real_X, tf.ones_like(real_X)), -1), training=False)
            identity_Y = self.generator_X2Y(tf.concat((real_Y, tf.ones_like(real_Y)), -1), training=False)

            d_real_X = self.discriminator_X(real_X, training=False)
            d_real_Y = self.discriminator_Y(real_Y, training=False)
            d_fake_X = self.discriminator_X(fake_X, training=False)
            d_fake_Y = self.discriminator_Y(fake_Y, training=False)

            d_real_X2 = self.discriminator_X2(real_X, training=False)
            d_real_Y2 = self.discriminator_Y2(real_Y, training=False)
            d_fake_X2 = self.discriminator_X2(cycle_X, training=False)
            d_fake_Y2 = self.discriminator_Y2(cycle_Y, training=False)


            loss_adv_X2Y, loss_adv2_X2Y, loss_cyc_X2Y, loss_id_X2Y= self.generator_loss(real_Y, cycle_Y, identity_Y, d_fake_Y, d_fake_Y2, self.test_batch_size)
            loss_adv_Y2X, loss_adv2_Y2X, loss_cyc_Y2X, loss_id_Y2X= self.generator_loss(real_X, cycle_X, identity_X, d_fake_X, d_fake_X2, self.test_batch_size)
            total_cycle_loss = self.lambda_cyc * (loss_cyc_X2Y + loss_cyc_Y2X)

            loss_X2Y = loss_adv_X2Y + loss_adv2_X2Y + total_cycle_loss + loss_id_X2Y
            loss_Y2X = loss_adv_Y2X + loss_adv2_Y2X + total_cycle_loss + loss_id_Y2X
                
            loss_X = self.discriminator_loss(d_real_X, d_fake_X, self.test_batch_size)
            loss_Y = self.discriminator_loss(d_real_Y, d_fake_Y, self.test_batch_size)

            loss_X2 = self.discriminator_loss(d_real_X2, d_fake_X2, self.test_batch_size)
            loss_Y2 = self.discriminator_loss(d_real_Y2, d_fake_Y2, self.test_batch_size)

            self.test_loss_X2Y.update_state(loss_X2Y * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_Y2X.update_state(loss_Y2X * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_X.update_state(loss_X * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_Y.update_state(loss_Y * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_X2.update_state(loss_X2 * self.distribute_strategy.num_replicas_in_sync)
            self.test_loss_Y2.update_state(loss_Y2 * self.distribute_strategy.num_replicas_in_sync)
 
            return real_X, real_Y, mask_X, mask_Y, fake_X, fake_Y, cycle_X, cycle_Y, identity_X, identity_Y

        return self.distribute_strategy.run(step_fn, args=(dataset_inputs,))


    def train(self):
        self.summary_writer = tf.summary.create_file_writer(logdir=self.log_dir)

        with self.summary_writer.as_default():  # 構造をテキストで保存
            # with io.StringIO() as buf:
            #     self.generator.summary(print_fn=lambda x: buf.write(x + "\n"))
            #     text = buf.getvalue()
            # tf.summary.text("generator_summary", text, 0)
            # with io.StringIO() as buf:
            #     self.discriminator.summary(print_fn=lambda x: buf.write(x + "\n"))
            #     text = buf.getvalue()
            # tf.summary.text("discriminator_summary", text, 0)
            tf.summary.text("args_summary", json.dumps(self.args_dict), 0)


        # with open(os.path.join(".", "datasets", "dataset_size.pkl"), 'rb') as p:
        #     train_size, test_size = pickle.load(p)  
        # print(f"base data size :: train : {train_size}, test : {test_size}")
        # train_size, test_size = int(train_size/self.train_batch_size * self.repeat_num), test_size//self.test_batch_size


        for iteration in tqdm(range(self.start_iteration, self.iterations), initial=self.start_iteration, total=self.iterations):

            ### train step ###
            # for i, train_data in enumerate(tqdm(self.train_dataset, leave=False, total=train_size)):
            for i, train_data in enumerate(self.train_dataset):

                if self.save_profile: 
                    if iteration == 0 and i == 3:
                        print("profiler start")
                        tf.profiler.experimental.start(self.log_dir)
                    if iteration == 1 and i == 3:
                        print("profiler stop")
                        tf.profiler.experimental.stop()

                if iteration == self.id_rate and i == 0:
                    print("change lambda id")
                    self.lambda_id = 0.0
                    # corresp for retracting
                    self._compile_learn_function()

                self.train_step_func(train_data)


            ### logging step ###
            # モデルの書き出し
            if (iteration + 1) % self.sample_interval == 0 or (iteration + 1) == self.iterations:
                save_dir = os.path.join(self.savedmodel_dir, f"{(iteration + 1):05d}")
                os.makedirs(save_dir, exist_ok=True)
                self.generator_X2Y.save_weights(os.path.join(save_dir, "X2Y.h5"))
                self.generator_Y2X.save_weights(os.path.join(save_dir, "Y2X.h5"))
            
            # # TensorBoardにloss類と元/生成melspecを保存
            if iteration % self.logging_interval == 0 or (iteration + 1) == self.iterations:
                
                ### test step ###
                for test_data in self.test_dataset:     
                    real_X, real_Y, mask_X, mask_Y, fake_X, fake_Y, cycle_X, cycle_Y, identity_X, identity_Y = self.test_step_func(test_data)

                
                if self.distribute_strategy.num_replicas_in_sync == 1:
                    melspecs = real_X[:5], real_Y[:5], mask_X[:5], mask_Y[:5], fake_X[:5], fake_Y[:5], cycle_X[:5], cycle_Y[:5], identity_X[:5], identity_Y[:5]
                else:
                    melspecs = real_X[0][:5], real_Y[0][:5], mask_X[0][:5], mask_Y[0][:5], fake_X[0][:5], fake_Y[0][:5], cycle_X[0][:5], cycle_Y[0][:5], identity_X[0][:5], identity_Y[0][:5]

                self._summary_write(iteration + 1, melspecs = melspecs)

            # 1epoch終了でreset
            self._reset_state()

            if (iteration + 1) % self.checkpoint_interval == 0 or (iteration + 1) == self.iterations:
                # 復元チェックポイント保存
                self.checkpoint_manager.save()
