import tensorflow as tf 
from tensorflow.keras.layers import Input, Conv2D,  LeakyReLU
from tensorflow.keras.layers import BatchNormalization, Conv2DTranspose



class UnetModel():
    def double_conv2d_down(self,entered_input, filters=64, dropout_prob=0.3, max_pooling=True):
        # Taking first input and implementing the first conv block
        conv1 = Conv2D(filters,3, padding = "same")(entered_input)
        batch_norm1 = BatchNormalization()(conv1)
        act1 = LeakyReLU(0.01)(batch_norm1)
        
        # Taking first input and implementing the second conv block
        conv2 = Conv2D(filters, kernel_size = (3,3), padding = "same")(act1)
        batch_norm2 = BatchNormalization()(conv2)
        act2 = LeakyReLU(0.01)(batch_norm2)

        if dropout_prob > 0:
            act2 = tf.keras.layers.Dropout(dropout_prob)(act2)
        if max_pooling:
            maxpool = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(act2)
        else:
            maxpool = act2
        
        sk_conn  = act2
        return maxpool,sk_conn

    def double_conv2d_up(self,entered_input,skip_layer_input, filters=64):
        # Taking first input and implementing the first conv block
        conv1 = Conv2DTranspose(filters,3, strides=(2,2), padding = "same")(entered_input)
        merge = tf.keras.layers.concatenate([conv1, skip_layer_input], axis=3)
        
        # Taking first input and implementing the second conv block
        conv2 = Conv2D(filters, kernel_size = (3,3),activation='relu', padding = "same")(merge)
        conv3 = Conv2D(filters, kernel_size = (3,3),activation='relu', padding = "same")(conv2)

        return conv3

    def UNetArch(self,input_size=(480,368,3), n_filters=64, n_classes=3):
        """
        Combine both encoder and decoder blocks according to the polar-Net research paper
        Return the model as output 
        """
        # Input size represent the size of 1 image (the size used for pre-processing) 
        inputs = Input(input_size)
        
        # Encoder includes multiple convolutional mini blocks with different maxpooling, dropout and filter parameters
        # Observe that the filters are increasing as we go deeper into the network which will increasse the # channels of the image 
        cblock1 = self.double_conv2d_down(inputs, n_filters,dropout_prob=0, max_pooling=True)
        cblock2 = self.double_conv2d_down(cblock1[0],n_filters*2,dropout_prob=0, max_pooling=True)
        cblock3 = self.double_conv2d_down(cblock2[0], n_filters*4,dropout_prob=0, max_pooling=True)
        cblock4 = self.double_conv2d_down(cblock3[0], n_filters*8,dropout_prob=0.3, max_pooling=True)
        cblock5 = self.double_conv2d_down(cblock4[0], n_filters*8, dropout_prob=0.3, max_pooling=False) 
        
        # Decoder includes multiple mini blocks with decreasing number of filters
        # Observe the skip connections from the encoder are given as input to the decoder
        # Recall the 2nd output of encoder block was skip connection, hence cblockn[1] is used
        ublock6 = self.double_conv2d_up(cblock5[0], cblock4[1],  n_filters * 8)
        ublock7 = self.double_conv2d_up(ublock6, cblock3[1],  n_filters * 4)
        ublock8 = self.double_conv2d_up(ublock7, cblock2[1],  n_filters * 2)
        ublock9 = self.double_conv2d_up(ublock8, cblock1[1],  n_filters)

        # Complete the model with 1 3x3 convolution layer (Same as the prev Conv Layers)
        # Followed by a 1x1 Conv layer to get the image to the desired size. 
        # Observe the number of channels will be equal to number of output classes
        conv9 = Conv2D(n_filters,
                    3,
                    activation='relu',
                    padding='same',
                    kernel_initializer='he_normal')(ublock9)

        conv10 = Conv2D(n_classes, 1, padding='same')(conv9)
        
        # Define the model
        model = tf.keras.Model(inputs=inputs, outputs=conv10)

        return model
