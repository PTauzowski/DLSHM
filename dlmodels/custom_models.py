import tensorflow as tf
from keras import Input, applications, initializers, layers, Model
from keras.layers import AveragePooling2D, Conv2D, BatchNormalization, UpSampling2D, Concatenate,Conv2DTranspose


def custom_vgg19( input_shape):
    model = tf.keras.applications.VGG19 (include_top=False, input_shape=input_shape, classes=3)
    x = model.layers[-1].output
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    outputs = Conv2DTranspose(8, (2, 2), strides=(2, 2), activation='sigmoid', padding='same')(x)
    model = Model(inputs=model.inputs, outputs=outputs)
    return model

def convolution_block(
    block_input,
    num_filters=256,
    kernel_size=3,
    dilation_rate=1,
    padding="same",
    use_bias=False,
):
    x = Conv2D(
        num_filters,
        kernel_size=kernel_size,
        dilation_rate=dilation_rate,
        padding="same",
        use_bias=use_bias,
        kernel_initializer=initializers.HeNormal(),
    )(block_input)
    x = BatchNormalization()(x)
    return layers.ReLU()(x)


def DilatedSpatialPyramidPooling(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_6 = convolution_block(dspp_input, kernel_size=3, dilation_rate=6)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_18 = convolution_block(dspp_input, kernel_size=3, dilation_rate=18)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes, output_activation='softmax'):
    model_input = Input(shape=(image_size[0], image_size[1], image_size[2]))
    resnet50 = applications.ResNet50(
        weights="imagenet", include_top=False, input_tensor=model_input
    )
    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPooling(x)

    input_a=Conv2DTranspose(48, (2, 2), strides=(4, 4), padding="same")(x)

    input_b = resnet50.get_layer("conv2_block3_2_relu").output
    input_b = convolution_block(input_b, num_filters=48, kernel_size=1)

    x = Concatenate(axis=-1)([input_a, input_b])
    x = convolution_block(x)
    x = convolution_block(x)
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding="same")(x)
    x = convolution_block(x)
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding="same")(x)
    model_output = Conv2D(num_classes, kernel_size=(1, 1), activation=output_activation, padding="same")(x)
    return Model(inputs=model_input, outputs=model_output)