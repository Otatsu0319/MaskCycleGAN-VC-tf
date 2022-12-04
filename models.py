import tensorflow as tf
import tensorflow_addons as tfa
import modules

class Generator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        # first block
        self.first_conv = tf.keras.layers.Conv2D(128, kernel_size = (5, 15), strides = (1, 1), padding="same")
        self.first_GLU = modules.GLU()

        # Downsample (2D)
        self.ds_l1 = modules.DownSampleBlock(256)
        self.ds_l2 = modules.DownSampleBlock(512)

        # 2D -> 1D
        self.to1d_conv = tf.keras.layers.Conv1D(256, kernel_size = 1, strides = 1, padding="same")
        self.to1d_IN = tfa.layers.InstanceNormalization()

        # 6 1D blocks
        self.block_1d_1 = modules.ResBlock(512)
        self.block_1d_2 = modules.ResBlock(512)
        self.block_1d_3 = modules.ResBlock(512)
        self.block_1d_4 = modules.ResBlock(512)
        self.block_1d_5 = modules.ResBlock(512)
        self.block_1d_6 = modules.ResBlock(512)

        # 1D -> 2D
        self.to2d_conv = tf.keras.layers.Conv1D(256 * 20, kernel_size = 1, strides = 1, padding="same")

        # Upsample (2D)
        self.us_l1 = modules.UpSampleBlock(1024)
        self.us_l2 = modules.UpSampleBlock(512)

        # last block
        self.last_conv = tf.keras.layers.Conv2D(80, kernel_size = (5, 15), strides = (1, 1), padding="same")
        
        self.output_conv = tf.keras.layers.Conv2D(1, kernel_size = (5, 5), strides = (1, 1), padding="same")
        self.last_linear = tf.keras.layers.Activation("linear", dtype="float32")
    
    def call(self, inputs):
        t_length = tf.keras.backend.int_shape(inputs)[2] # (batch:64, h:80, t_length:256, c:1)

        # first block
        x = self.first_conv(inputs)
        x = self.first_GLU(x)

        # Downsample (2D)
        x = self.ds_l1(x)
        x = self.ds_l2(x)
        
        # 2D -> 1D
        x = tf.transpose(x, perm=[0, 3, 1, 2]) # NHWC -> NCHW
        x = tf.reshape(x, (-1, 256 * 20, 1, t_length//4))
        x = tf.squeeze(x, [2])
        x = tf.transpose(x, perm=[0, 2, 1]) # NCH -> NHC
        x = self.to1d_conv(x)
        x = self.to1d_IN(x)

        # 6 1D blocks
        x = self.block_1d_1(x)
        x = self.block_1d_2(x)
        x = self.block_1d_3(x)
        x = self.block_1d_4(x)
        x = self.block_1d_5(x)
        x = self.block_1d_6(x)

        # 1D -> 2D
        x = self.to2d_conv(x)
        x = tf.transpose(x, perm=[0, 2, 1]) # NHC -> NCH
        x = tf.reshape(x, (-1, 256, 20, t_length//4)) 
        x = tf.transpose(x, perm=[0, 2, 3, 1]) # NCHW -> NHWC

        # Upsample (2D)
        x = self.us_l1(x)
        x = self.us_l2(x)

        # last block
        x = self.last_conv(x)
        x = tf.transpose(x, perm=[0, 3, 2, 1])
        x = self.output_conv(x)
        x = self.last_linear(x)

        return x

    def model(self, x):
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    
class Discriminator(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.first_conv = tf.keras.layers.Conv2D(128, kernel_size = (3, 3), strides = (1, 1), padding="same")
        self.first_glu = modules.GLU()

        # Downsample (2D)
        self.ds_l1 = modules.DownSampleBlock(256, kernel_size = (3, 3))
        self.ds_l2 = modules.DownSampleBlock(512, kernel_size = (3, 3))
        self.ds_l3 = modules.DownSampleBlock(1024, kernel_size = (3, 3))
        self.ds_l4 = modules.DownSampleBlock(1024, kernel_size = (1, 5), strides = (1, 1), padding="same")

        self.last_conv = tf.keras.layers.Conv2D(1, kernel_size = (1, 3), strides = (1, 1), padding="same")
        self.last_activation = tf.keras.layers.Activation("sigmoid", dtype="float32")
    
    def call(self, inputs):
        x = self.first_conv(inputs)
        x = self.first_glu(x)

        # Downsample (2D)
        x = self.ds_l1(x)
        x = self.ds_l2(x)
        x = self.ds_l3(x)
        x = self.ds_l4(x)

        x = self.last_conv(x)
        x = self.last_activation(x)

        return x
    
    def model(self, x):
        return tf.keras.Model(inputs=[x], outputs=self.call(x))


if __name__ == "__main__":
    generator = Generator()
    x = tf.keras.layers.Input(shape=(80, 1024, 2))
    print(generator.model(x).summary())

    discriminator = Discriminator()
    # discriminator(dummy_input)
    print(discriminator.model(x).summary())
    
    generator = Generator()
    dummy_input = tf.random.uniform((128, 80, 128, 1))
    dummy_result = generator(dummy_input)
    print(dummy_result.shape)