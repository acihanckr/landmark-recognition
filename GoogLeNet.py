import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Sequential, Flatten, Dense, Softmax, MaxPooling2D, Concatenate

## still in progress

# GoogLeNet version of Inception "Going deeper with convolutions"
def Inception(x, n_conv1x1, n_reduce3x3, n_conv3x3, n_reduce5x5, n_conv5x5, n_pool_proj):
    branch1_a = Conv2D(n_conv1x1, (1, 1), padding='same', activation='relu')
    
    branch2_a = Conv2D(n_reduce3x3, (1, 1), padding='same', activation='relu')
    branch2_b = Conv2D(n_conv3x3, (3, 3), padding='same', activation='relu')
    
    branch3_a = Conv2D(n_reduce5x5, (1, 1), padding='same', activation='relu')
    branch3_b = Conv2D(n_conv5x5, (5, 5), padding='same', activation='relu')
    
    branch4_a = MaxPooling2D((3, 3))(x)
    branch4_b = Conv2D(n_pool_proj, (1, 1), padding='same', activation='relu')
    
    depth_concat = Concatenate(axis=-1)
    
    def call(x):
        z1 = branch1_a(x)
        z2 = branch2_b(branch2_a(x))
        z3 = branch3_b(branch3_a(x))
        z4 = branch4_b(branch4_a(x))
        return depth_concat([z1, z2, z3, z4])
    return call

input_shape = (224, 224, 3)

class GoogLeNet(tf.keras.Model):
    def ___init__(self):
        super(tf.keras.Model, self).__init__()
        self.conv1 = Conv2D(64,
            kernel_size=(7,7), strides=(2), 
            activation = 'relu', padding = 'SAME', 
            input_shape = input_shape, data_format = 'channels_last')
        self.pool1 = None
        self.conv2 = Conv2D(192, 
            kernel_size=(3,3), strides=(1), 
            activation = 'relu', padding = 'SAME', 
            input_shape = input_shape, data_format = 'channels_last')
        self.pool2 = None
        self.inception3a = Inception(64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(128, 128, 192, 32, 96, 64)
        self.pool3 = None
        self.inception4a = Inception(192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(160, 112,224, 24, 64, 64)
        self.inception4c = Inception(128, 128,256, 24, 64, 64)
        self.inception4d = Inception(112, 144,288, 32, 64, 64)
        self.inception4e = Inception(256, 160,320, 32, 128,128)
        self.pool3 = None
        self.inception5a = None
        self.inception5b = None
        self.pool4 = None
        self.dropout = Dropout(p=0.4)
        self.d1
        
    def call(self, x):
        pass