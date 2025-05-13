# tracknet.py
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, Activation, MaxPooling2D, UpSampling2D, Reshape, Permute

def trackNet(n_classes, input_height=360, input_width=640):
    """
    TrackNet模型定义 - 用于网球追踪
    参数:
        n_classes: 输出类别数（通常为256）
        input_height: 输入图像高度（默认360）
        input_width: 输入图像宽度（默认640）
    """
    # 输入层，注意channels_first格式
    imgs_input = Input(shape=(3, input_height, input_width))

    # Layer 1
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(imgs_input)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 2
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 3
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

    # Layer 4
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 5
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 6
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

    # Layer 7
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 8
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 9
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 10
    x = MaxPooling2D((2, 2), strides=(2, 2), data_format='channels_first')(x)

    # Layer 11
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 12
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 13
    x = Conv2D(512, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 14
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # Layer 15
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 16
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 17
    x = Conv2D(256, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 18
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # Layer 19
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 20
    x = Conv2D(128, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 21
    x = UpSampling2D((2, 2), data_format='channels_first')(x)

    # Layer 22
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 23
    x = Conv2D(64, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    # Layer 24
    x = Conv2D(n_classes, (3, 3), kernel_initializer='random_uniform', padding='same', data_format='channels_first')(x)
    x = Activation('relu')(x)
    x = BatchNormalization()(x)

    o_shape = Model(imgs_input, x).output_shape
    print("layer24 output shape:", o_shape[1], o_shape[2], o_shape[3])
    
    OutputHeight = o_shape[2]
    OutputWidth = o_shape[3]

    # Reshape the size to (256, 360*640)
    x = Reshape((-1, OutputHeight*OutputWidth))(x)

    # Change dimension order to (360*640, 256)
    x = Permute((2, 1))(x)

    # Layer 25
    gaussian_output = Activation('softmax')(x)

    model = Model(imgs_input, gaussian_output)
    model.outputWidth = OutputWidth
    model.outputHeight = OutputHeight

    # Show model's details
    model.summary()

    return model 