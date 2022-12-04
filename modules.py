import tensorflow as tf
import tensorflow_addons as tfa

class GLU(tf.keras.layers.Layer):
    '''
        I imitated Torch's implementation. Halve the number of channels.
        https://pytorch.org/docs/stable/generated/torch.nn.GLU.html
    '''
    def __init__(self, axis = -1, **kwargs):
        self.axis = axis
        super(GLU, self).__init__(**kwargs)

    def call(self, inputs): 
        a, b = tf.split(inputs, num_or_size_splits=2, axis=self.axis)
        return tf.multiply(a, tf.keras.activations.sigmoid(b))


class PixelShuffler(tf.keras.layers.Layer):
    def __init__(self, r = 2, **kwargs):
        self.r = r
        super().__init__(**kwargs)

    def call(self, inputs): 
        return tf.nn.depth_to_space(inputs, block_size=self.r)


class DownSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, kernel_size = (5, 5), strides = (2, 2), padding="same", **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filter_size, kernel_size = kernel_size, strides = strides, padding=padding)
        self.IN = tfa.layers.InstanceNormalization()
        self.GLU = GLU()
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.IN(x)
        x = self.GLU(x)
        return x


class UpSampleBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv = tf.keras.layers.Conv2D(filter_size, kernel_size = (5, 5), strides = (1, 1), padding="same")
        self.PS = PixelShuffler(r = 2)
        self.IN = tfa.layers.InstanceNormalization()
        self.GLU = GLU()
    
    def call(self, inputs):
        x = self.conv(inputs)
        x = self.PS(x)
        x = self.IN(x)
        x = self.GLU(x)
        return x


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, filter_size, **kwargs):
        super().__init__(**kwargs)
        self.conv_1 = tf.keras.layers.Conv1D(filter_size, kernel_size = 3, strides = 1, padding="same")
        self.IN_1 = tfa.layers.InstanceNormalization()
        self.GLU = GLU()
        self.conv_2 = tf.keras.layers.Conv1D(filter_size//2, kernel_size = 3, strides = 1, padding="same")
        self.IN_2 = tfa.layers.InstanceNormalization()
        self.sum = tf.keras.layers.Add()

    def call(self, inputs):
        x = self.conv_1(inputs)
        x = self.IN_1(x)
        x = self.GLU(x)
        x = self.conv_2(x)
        x = self.IN_2(x)
        x = self.sum([x, inputs])
        return x
