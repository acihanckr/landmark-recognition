import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Softmax, MaxPooling2D, Concatenate, Dropout, Lambda, Input, ZeroPadding2D
from tensorflow.keras import Model
from tensorflow.python.keras.layers.pooling import AveragePooling2D

## still in progress

# GoogLeNet version of Inception "Going deeper with convolutions"
def Inception(n_conv1x1, n_reduce3x3, n_conv3x3, n_reduce5x5, n_conv5x5, n_pool_proj):
    branch1_a = Conv2D(n_conv1x1, (1, 1), padding='same', activation='relu')
    
    branch2_a = Conv2D(n_reduce3x3, (1, 1), padding='same', activation='relu')
    zero1 = ZeroPadding2D(padding=(1, 1))
    branch2_b = Conv2D(n_conv3x3, (3, 3), padding='valid', activation='relu')
    
    branch3_a = Conv2D(n_reduce5x5, (1, 1), padding='same', activation='relu')
    zero2 = ZeroPadding2D(padding=(2, 2))
    branch3_b = Conv2D(n_conv5x5, (5, 5), padding='valid', activation='relu')
    
    branch4_a = MaxPooling2D((3, 3), strides=(1), padding='same')
    branch4_b = Conv2D(n_pool_proj, (1, 1), padding='same', activation='relu')
    
    depth_concat = Concatenate(axis=-1)
    
    def call(x):
        z1 = branch1_a(x)
        z2 = branch2_b(zero1(branch2_a(x)))
        z3 = branch3_b(zero2(branch3_a(x)))
        z4 = branch4_b(branch4_a(x))
        return depth_concat([z1, z2, z3, z4])
    return call

LRN = lambda: Lambda(tf.nn.local_response_normalization)

class GoogLeNet():
    @staticmethod
    def create_model(input_shape, num_classes):
        
        conv1 = Conv2D(64,
            kernel_size=(7,7), strides=(2), 
            activation = 'relu', padding = 'SAME',
            data_format = 'channels_last')
        pool1 = MaxPooling2D((3, 3), strides=(2))
        lrn1 = LRN()

        conv2a = Conv2D(64, 
            kernel_size=(1, 1),
            activation = 'relu', padding = 'SAME',
            data_format = 'channels_last')
        conv2 = Conv2D(192, 
            kernel_size=(3,3), strides=(1), 
            activation = 'relu', padding = 'SAME',
            data_format = 'channels_last')

        lrn2 = LRN()

        pool2 = MaxPooling2D((3, 3), strides=(2))

        inception3a = Inception(64, 96, 128, 16, 32, 32)
        inception3b = Inception(128, 128, 192, 32, 96, 64)

        pool3 = MaxPooling2D((3, 3), strides=(2))

        inception4a = Inception(192, 96, 208, 16, 48, 64)
        inception4b = Inception(160, 112,224, 24, 64, 64)
        inception4c = Inception(128, 128,256, 24, 64, 64)
        inception4d = Inception(112, 144,288, 32, 64, 64)
        inception4e = Inception(256, 160,320, 32, 128,128)

        pool4 = MaxPooling2D((3, 3), strides=(2))

        inception5a = Inception(256, 160,320, 32, 128,128)
        inception5b = Inception(384, 192,384, 48, 128,128)


        # first branch
        out1_pool = AveragePooling2D((5, 5), strides=(3))
        out1_conv = Conv2D(128, 
            kernel_size=(1, 1),
            activation = 'relu', padding = 'SAME',
            data_format = 'channels_last')
        out1_flat = Flatten()
        out1_dense1 = Dense(1024, activation="relu")
        out1_dropout = Dropout(rate=0.7)
        out1_dense2 = Dense(num_classes)
        out1_softmax = Softmax()

        # second branch
        out2_pool = AveragePooling2D((5, 5), strides=(3))
        out2_conv = Conv2D(128, 
            kernel_size=(1, 1),
            activation = 'relu', padding = 'SAME',
            data_format = 'channels_last')
        out2_flat = Flatten()
        out2_dense1 = Dense(1024, activation="relu")
        out2_dropout = Dropout(rate=0.7)
        out2_dense2 = Dense(num_classes)
        out2_softmax = Softmax()

        # third branch
        out3_pool = AveragePooling2D((7, 7), strides=(1))
        out3_flat = Flatten()
        out3_dropout = Dropout(rate=0.4)
        out3_dense = Dense(num_classes)
        out3_softmax = Softmax()


        ### Model
        inputs = Input(shape=input_shape)
        z = conv1(inputs)
        z = pool1(z)
        z = lrn1(z)
        z = conv2a(z)
        z = conv2(z)
        z = lrn2(z)
        z = pool2(z)
        z = inception3a(z)
        z = inception3b(z)
        z = pool3(z)
        z = inception4a(z)

        # first branch
        z1 = out1_pool(z)
        z1 = out1_conv(z1)
        z1 = out1_flat(z1)
        z1 = out1_dense1(z1)
        z1 = out1_dropout(z1)
        z1 = out1_dense2(z1)
        z1 = out1_softmax(z1)

        z = inception4b(z)
        z = inception4c(z)
        z = inception4d(z)

        # second branch
        z2 = out2_pool(z)
        z2 = out2_conv(z2)
        z2 = out2_flat(z2)
        z2 = out2_dense1(z2)
        z2 = out2_dropout(z2)
        z2 = out2_dense2(z2)
        z2 = out2_softmax(z2)

        z = inception4e(z)
        z = pool4(z)
        z = inception5a(z)
        z = inception5b(z)

        # third branch
        z3 = out3_pool(z)
        z3 = out3_flat(z3)
        z3 = out3_dropout(z3)
        z3 = out3_dense(z3)
        z3 = out3_softmax(z3)

        outputs = [z1, z2, z3]

        return Model(input=inputs, outputs=outputs)