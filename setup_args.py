import datetime
class Args:
    def __init__(self):
        self.mixed_precision = True
        self.use_tpu = False
        self.num_workers = 1
        
        
        self.mel_size = 80
        self.sampling_rate = 22050
        self.fmin = 80
        self.fmax = 7600
        self.trim_threshold_in_db = 60
        self.trim_frame_size = 2048
        self.trim_hop_size = 512
        self.frame_period = 5.0 # ms
        self.fft_size = 1024
        self.hop_size = 256
        self.win_length = None
        self.window = "hann"
        self.preprocess_thread_num = 24
        
        self.using_jvs_id_x = "jvs009"
        self.using_jvs_id_y = "jvs002"
        self.train_data_ratio = 0.70
        self.preset_datafile_ratio = 0.5
        self.remake_datasets = False
        self.shuffle_buffer_size = 100

        self.repeat_num = 8
        self.iterations = 5*(10**5) // self.repeat_num
        self.train_batch_size = 64 # org:1
        self.test_batch_size = 4
        self.dataset_t_length = 64 # 0.75 sec
        self.dataset_hop_size = 32
        self.logging_interval = 5000 // self.repeat_num
        self.sample_interval = 10000 // self.repeat_num
        self.checkpoint_interval = 5000 // self.repeat_num
        self.print_log = False
        
        self.mask_mode = "FIF" # FIF(consecutive frames), FIF_ns(discontinuous frame), FIS(Mask spectral bands randomly), FIP(Mask Points (pixels? Similar to Dropout))
        self.mask_percent = 50 # [%], mask_size = dataset_t_length * mask_size / 100
        self.mask_size = int(self.dataset_t_length * self.mask_percent / 100) # mask_size = dataset_t_length * mask_percent / 100
        # mask start point : randomly determined within the range of [0, mask_size[%]]
        
        self.g_learn_rate = 0.0002
        self.d_learn_rate = 0.0001
        self.beta_1 = 0.5
        self.beta_2 = 0.999

        self.lambda_cyc = 10.0
        self.lambda_id = 5.0
        self.id_rate = 10**4 // self.repeat_num

        self.model_name = "maskcyclegan_vc2"

        self.tensorboard_log_dir = "logs"
        self.save_profile = False

        self.restore_bool = True
        if self.restore_bool:
            self.start_iteration = 25625
            self.datetime = "20221205-045950"
            
            if self.id_rate <= self.start_iteration:
                self.lambda_id = 0.0
                print("change lambda id")
        else:
            self.start_iteration = 0
            now = datetime.datetime.utcnow() + datetime.timedelta(hours=9)
            self.datetime = now.strftime("%Y%m%d-%H%M%S")

        self.set_seed = True
        self.seed = 2050

if __name__ == "__main__":
    args = Args()
    print(args.__dict__)