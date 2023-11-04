from keras.layers import *
from keras.models import *

from keras.layers import *
import tensorflow as tf


def expend_as(tensor, rep):
    my_repeat = Lambda(lambda x: tf.repeat(x, rep, axis=3))(tensor)
    return my_repeat


# Channel attentation
def Channelblock(up_data, skip_data, filters):
    conv1 = Conv2D(filters, (1, 1), padding="same")(up_data)
    batch1 = BatchNormalization()(conv1)
    up_LeakyReLU1 = ReLU()(batch1)

    # up_LeakyReLU1 = Conv2D(filters, (3, 3), padding="same")(up_data)
    # up_LeakyReLU1 = BatchNormalization()(up_LeakyReLU1)
    # up_LeakyReLU1 = ReLU()(up_LeakyReLU1)

    # conv2 = Conv2D(filters, (1, 1), padding="same")(skip_data)
    # batch2 = BatchNormalization()(conv2)
    # skip_LeakyReLU2 = ReLU()(batch2)
    skip_LeakyReLU2 = Conv2D(filters, (3, 3), padding="same")(skip_data)
    skip_LeakyReLU2 = BatchNormalization()(skip_LeakyReLU2)
    skip_LeakyReLU2 = ReLU()(skip_LeakyReLU2)

    data3 = concatenate([up_LeakyReLU1, skip_LeakyReLU2])

    # Channel self-attention
    data3 = GlobalAveragePooling2D()(data3)
    data3 = Dense(units=filters)(data3)
    data3 = BatchNormalization()(data3)
    data3 = ReLU()(data3)
    data3 = Dense(units=filters)(data3)
    data3 = Activation('sigmoid')(data3)

    a = Reshape((1, 1, filters))(data3)

    a1 = 1 - data3
    a1 = Reshape((1, 1, filters))(a1)

    up_Cout = multiply([up_LeakyReLU1, a])

    skip_Cout = multiply([skip_LeakyReLU2, a1])

    return up_Cout, skip_Cout


# spatial self-attention
def Spatialblock(up_data, skip_data, filters):
    conv1 = Conv2D(filters, (1, 1), padding="same")(up_data)
    batch1 = BatchNormalization()(conv1)
    up_LeakyReLU1 = ReLU()(batch1)

    conv2 = Conv2D(filters, (3, 3), padding="same")(skip_data)
    batch2 = BatchNormalization()(conv2)
    skip_LeakyReLU2 = ReLU()(batch2)

    data3 = add([up_LeakyReLU1, skip_LeakyReLU2])
    data3 = ReLU()(data3)
    data3 = Conv2D(1, (1, 1), padding='same')(data3)
    data3 = Activation('sigmoid')(data3)

    a = expend_as(data3, filters)
    up_Sout = multiply([a, up_LeakyReLU1])

    a1 = 1 - data3
    a1 = expend_as(a1, filters)
    skip_Sout = multiply([a1, skip_LeakyReLU2])

    return up_Sout, skip_Sout


def PSAM(up_data, skip_data, filters):
    x1, x2 = Channelblock(up_data, skip_data, filters)
    y1, y2 = Spatialblock(up_data, skip_data, filters)
    out_data = Concatenate()([x1, x2, y1, y2])
    out_data = Conv2D(filters, (1, 1), padding="same")(out_data)
    out_data = BatchNormalization()(out_data)
    out_data = LeakyReLU(alpha=0.01)(out_data)
    return out_data


def Cross_ASPP(x, filters):
    shape = x.shape

    y1 = AveragePooling2D(pool_size=(shape[1], shape[2]))(x)
    y1 = Conv2D(filters, 1, padding="same")(y1)
    y1 = BatchNormalization()(y1)
    y1 = Activation("relu")(y1)
    y1 = UpSampling2D((shape[1], shape[2]), interpolation='bilinear')(y1)

    y2 = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(x)
    y2 = BatchNormalization()(y2)
    y2 = Activation("relu")(y2)

    y3 = add([x, y2])
    y3 = Conv2D(filters, 3, dilation_rate=6, padding="same", use_bias=False)(y3)
    y3 = BatchNormalization()(y3)
    y3 = Activation("relu")(y3)

    y4 = add([x, y3])
    y4 = Conv2D(filters, 3, dilation_rate=12, padding="same", use_bias=False)(y4)
    y4 = BatchNormalization()(y4)
    y4 = Activation("relu")(y4)

    y5 = add([x, y4])
    y5 = Conv2D(filters, 3, dilation_rate=18, padding="same", use_bias=False)(y5)
    y5 = BatchNormalization()(y5)
    y5 = Activation("relu")(y5)

    y = Concatenate()([y1, y2, y3, y4, y5])

    y = Conv2D(filters, 1, dilation_rate=1, padding="same", use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Activation("relu")(y)

    return y


def ConvBlock(data, filte):
    conv1 = Conv2D(filte, (3, 3), padding="same")(data)  # ,dilation_rate=(4,4)
    batch1 = BatchNormalization()(conv1)
    LeakyReLU1 = LeakyReLU(alpha=0.01)(batch1)
    conv2 = Conv2D(filte, (3, 3), padding="same")(LeakyReLU1)
    batch2 = BatchNormalization()(conv2)
    LeakyReLU2 = LeakyReLU(alpha=0.01)(batch2)
    return LeakyReLU2


def squeeze_excite_block(inputs, ratio=8):
    init = inputs
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = GlobalAveragePooling2D()(init)
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = Multiply()([init, se])
    return x


def SE_Conv(data, filte):
    x = data
    shortcut = data

    x = Conv2D(filte, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filte, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filte, (1, 1), padding="same")(shortcut)
    shortcut = BatchNormalization()(shortcut)

    x = add([shortcut, x])
    x = Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def SKupdataplus(filters, data, skip_data):
    up_data = UpSampling2D((2, 2))(data)
    skip_data0 = PSAM(up_data, skip_data, filters)
    selective_data = Concatenate()([skip_data0, up_data])
    conv1 = ConvBlock(data=selective_data, filte=filters)
    # conv2 = ConvBlock(data=conv1, filte=filters)
    return conv1


def build(shape):
    inputs = Input(shape)
    Conv1 = SE_Conv(data=inputs, filte=32)
    Conv1 = SE_Conv(data=Conv1, filte=32)

    pool1 = MaxPooling2D(pool_size=(2, 2))(Conv1)
    Conv2 = SE_Conv(data=pool1, filte=64)
    Conv2 = SE_Conv(data=Conv2, filte=64)

    pool2 = MaxPooling2D(pool_size=(2, 2))(Conv2)
    Conv3 = SE_Conv(data=pool2, filte=128)
    Conv3 = SE_Conv(data=Conv3, filte=128)

    pool3 = MaxPooling2D(pool_size=(2, 2))(Conv3)
    Conv4 = SE_Conv(data=pool3, filte=256)
    Conv4 = SE_Conv(data=Conv4, filte=256)

    pool4 = MaxPooling2D(pool_size=(2, 2))(Conv4)
    Conv5 = SE_Conv(data=pool4, filte=512)
    Conv5 = SE_Conv(data=Conv5, filte=512)
    Conv5 = Cross_ASPP(Conv5, 512)

    up1 = SKupdataplus(filters=256, data=Conv5, skip_data=Conv4)

    up2 = SKupdataplus(filters=128, data=up1, skip_data=Conv3)

    up3 = SKupdataplus(filters=64, data=up2, skip_data=Conv2)

    up4 = SKupdataplus(filters=32, data=up3, skip_data=Conv1)

    outconv = Conv2D(1, (1, 1), strides=(1, 1), padding='same')(up4)
    out1 = Activation('sigmoid', name='out1')(outconv)

    model = Model(inputs=inputs, outputs=out1)
    return model


if __name__ == "__main__":
    shape = (352, 352, 3)
    model = build(shape)
    model.summary()
