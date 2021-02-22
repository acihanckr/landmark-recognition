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

def sequence(fns):

    def inner(z):
        res = z
        for f in fns:
            res = f(res)
        return res

    return inner

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
        first_layers = sequence([
            conv1, pool1, lrn1, conv2a, conv2, lrn2, pool2, inception3a, inception3b, pool3, inception4a
        ])
        curr = first_layers(inputs)

        # first branch
        first_branch = sequence([
            out1_pool, out1_conv, out1_flat, out1_dense1, out1_dropout, out1_dense2, out1_softmax
        ])
        out1 = first_branch(curr)

        mid_layers_1 = sequence([inception4b, inception4c, inception4d])
        curr = mid_layers_1(curr)

        # second branch
        second_branch = sequence([
            out2_pool, out2_conv, out2_flat, out2_dense1, out2_dropout, out2_dense2, out2_softmax
        ])
        out2 = second_branch(curr)

        mid_layers_2 = sequence([inception4e, pool4, inception5a, inception5b])
        curr = mid_layers_2(curr)

        # third branch
        third_branch = sequence([
            out3_pool, out3_flat, out3_dropout, out3_dense, out3_softmax
        ])
        out3 = third_branch(curr)

        return Model(input=inputs, outputs=[out1, out2, out3])