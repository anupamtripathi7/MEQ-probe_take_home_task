from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, BatchNormalization, Conv1D, Conv1DTranspose
from tensorflow.keras.layers import MaxPooling2D, MaxPooling1D
from tensorflow.keras.layers import concatenate, ZeroPadding1D


def build_2d_unet(img_height, img_width, channels):

    inputs = Input((img_height, img_width, channels))

    c0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c0 = BatchNormalization()(c0)
    c0 = Dropout(0.1)(c0)
    c0 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c0)
    c0 = BatchNormalization()(c0)
    p0 = MaxPooling2D((2, 2))(c0)

    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p0)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    u10 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c9)
    u10 = concatenate([u10, c0], axis=3)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    c10 = BatchNormalization()(c10)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c10)

    return Model(inputs=[inputs], outputs=[outputs])

def check_and_add_padding(x, y):
    if y.shape[1] % 2 != 0:
        return ZeroPadding1D(padding=(1, 0))(x)
    return x


def build_1d_unet(signal_length, channels):
    inputs = Input((signal_length, channels))

    c0 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    c0 = BatchNormalization()(c0)
    c0 = Dropout(0.1)(c0)
    c0 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c0)
    c0 = BatchNormalization()(c0)
    p0 = MaxPooling1D(2)(c0)

    c1 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(p0)
    c1 = BatchNormalization()(c1)
    c1 = Dropout(0.1)(c1)
    c1 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling1D(2)(c1)

    c2 = Conv1D(64, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    c2 = Dropout(0.1)(c2)
    c2 = Conv1D(64, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling1D(2)(c2)

    c3 = Conv1D(128, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = BatchNormalization()(c3)
    c3 = Dropout(0.2)(c3)
    c3 = Conv1D(128, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    c3 = BatchNormalization()(c3)
    p3 = MaxPooling1D(2)(c3)

    c4 = Conv1D(256, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = BatchNormalization()(c4)
    c4 = Dropout(0.2)(c4)
    c4 = Conv1D(256, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    c4 = BatchNormalization()(c4)
    p4 = MaxPooling1D(2)(c4)

    c5 = Conv1D(512, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = BatchNormalization()(c5)
    c5 = Dropout(0.3)(c5)
    c5 = Conv1D(512, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c5)
    c5 = BatchNormalization()(c5)

    u6 = Conv1DTranspose(128, (2,), strides=(2,), padding='same')(c5)
    u6 = check_and_add_padding(u6, c4)
    u6 = concatenate([u6, c4])
    c6 = Conv1D(256, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = BatchNormalization()(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv1D(256, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c6)
    c6 = BatchNormalization()(c6)

    u7 = Conv1DTranspose(64, (2,), strides=(2,), padding='same')(c6)
    u7 = check_and_add_padding(u7, c3)
    u7 = concatenate([u7, c3])
    c7 = Conv1D(128, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = BatchNormalization()(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv1D(128, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c7)
    c7 = BatchNormalization()(c7)

    u8 = Conv1DTranspose(32, (2,), strides=(2,), padding='same')(c7)
    u8 = check_and_add_padding(u8, c2)
    u8 = concatenate([u8, c2])
    c8 = Conv1D(64, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = BatchNormalization()(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv1D(64, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c8)
    c8 = BatchNormalization()(c8)

    u9 = Conv1DTranspose(16, (2,), strides=(2,), padding='same')(c8)
    u9 = check_and_add_padding(u9, c1)
    u9 = concatenate([u9, c1])
    c9 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = BatchNormalization()(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c9)
    c9 = BatchNormalization()(c9)

    u10 = Conv1DTranspose(16, (2,), strides=(2,), padding='same')(c9)
    u10 = check_and_add_padding(u10, c0)
    u10 = concatenate([u10, c0])
    c10 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(u10)
    c10 = BatchNormalization()(c10)
    c10 = Dropout(0.1)(c10)
    c10 = Conv1D(32, (3,), activation='relu', kernel_initializer='he_normal', padding='same')(c10)
    c10 = BatchNormalization()(c10)

    outputs = Conv1D(1, (1,), activation='sigmoid')(c10)

    return Model(inputs=[inputs], outputs=[outputs])


if __name__ == '__main__':
    unet_2d = build_2d_unet(img_height=512, img_width=512, channels=3)
    unet_2d.summary()

    # # Task 1: Build a 1D UNet for a signal length 1024
    unet_1d_1024 = build_1d_unet(signal_length=1024, channels=9)
    unet_1d_1024.summary()

#     # Task 2: Extend the code to work for signals that aren't an even power of 2
    unet_1d_1400 = build_1d_unet(signal_length=1400, channels=9)
    unet_1d_1400.summary()
