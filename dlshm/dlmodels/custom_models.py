import tensorflow as tf
from keras import Input, applications, initializers, layers, Model
from keras.layers import AveragePooling2D, Conv2D, BatchNormalization, UpSampling2D, Concatenate,Conv2DTranspose

import tensorflow as tf
from tensorflow import keras
from keras import layers


def conv_bn_relu(filters, kernel_size, strides=1, dilation_rate=1):
    def layer(x):
        x = layers.Conv2D(filters, kernel_size, strides=strides, padding="same",
                          dilation_rate=dilation_rate, use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.ReLU()(x)

    return layer


def ASPP1(x, filters=256):
    """ Atrous Spatial Pyramid Pooling """
    pool = layers.GlobalAveragePooling2D()(x)
    pool = layers.Reshape((1, 1, x.shape[-1]))(pool)
    pool = layers.Conv2D(filters, 1, padding="same", use_bias=False)(pool)
    pool = layers.BatchNormalization()(pool)
    pool = layers.ReLU()(pool)
    pool = layers.UpSampling2D(size=(x.shape[1], x.shape[2]), interpolation="bilinear")(pool)

    conv_1x1 = conv_bn_relu(filters, 1)(x)
    conv_3x3_1 = conv_bn_relu(filters, 3, dilation_rate=6)(x)
    conv_3x3_2 = conv_bn_relu(filters, 3, dilation_rate=12)(x)
    conv_3x3_3 = conv_bn_relu(filters, 3, dilation_rate=18)(x)

    x = layers.Concatenate()([pool, conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3])
    x = conv_bn_relu(filters, 1)(x)
    return x

def ASPP2(x):
    """ Atrous Spatial Pyramid Pooling module """
    conv1 = layers.Conv2D(256, 1, padding="same", activation="relu")(x)
    conv3 = layers.Conv2D(256, 3, dilation_rate=6, padding="same", activation="relu")(x)
    conv5 = layers.Conv2D(256, 3, dilation_rate=12, padding="same", activation="relu")(x)
    conv7 = layers.Conv2D(256, 3, dilation_rate=18, padding="same", activation="relu")(x)
    concat = layers.Concatenate()([conv1, conv3, conv5, conv7])
    return layers.Conv2D(256, 1, padding="same", activation="relu")(concat)



def DeepLabV3_2(input_shape=(512, 512, 3), num_classes=21, backbone='resnet50'):
    """ DeepLab v3 implementation with ResNet backbone. """
    base_model = keras.applications.ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.get_layer("conv4_block6_out").output  # Output of ResNet block 4
    x = ASPP2(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = layers.Conv2D(num_classes, 1, padding="same")(x)

    # Apply softmax (multi-class) or sigmoid (binary segmentation)
    if num_classes > 1:
        x = layers.Activation("softmax")(x)  # Multi-class probability output
    else:
        x = layers.Activation("sigmoid")(x)  # Binary segmentation probability output

    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    return keras.Model(inputs=base_model.input, outputs=x)

def DeepLabV3_1(input_shape=(512, 512, 3), num_classes=21, backbone='resnet50'):
    """ DeepLab v3 implementation with ResNet backbone. """
    base_model = keras.applications.ResNet101(weights="imagenet", include_top=False, input_shape=input_shape)
    x = base_model.get_layer("conv4_block6_out").output  # Output of ResNet block 4
    x = ASPP1(x)
    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)
    x = layers.Conv2D(num_classes, 1, padding="same")(x)

    # Apply softmax (multi-class) or sigmoid (binary segmentation)
    if num_classes > 1:
        x = layers.Activation("softmax")(x)  # Multi-class probability output
    else:
        x = layers.Activation("sigmoid")(x)  # Binary segmentation probability output

    x = layers.UpSampling2D(size=(4, 4), interpolation="bilinear")(x)

    return keras.Model(inputs=base_model.input, outputs=x)





def build_vgg19_segmentation_model(input_shape, num_classes=8):
    # Use VGG19 without the top layers
    vgg19 = tf.keras.applications.VGG19(weights="imagenet",include_top=False, input_shape=input_shape)

    # Freeze the VGG19 layers
    for layer in vgg19.layers:
        layer.trainable = False

    # Input layer
    inputs = layers.Input(shape=input_shape)

    # Use VGG19 as a feature extractor
    x = vgg19(inputs)

    # Add custom layers on top for segmentation
    x = layers.Conv2D(1024, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(4, 4))(x)  # Upsample to match the original image size
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)  # Upsample to match the original image size
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.UpSampling2D(size=(2, 2))(x)
    x = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(x)

    # Final segmentation layer
    outputs = layers.Conv2D(num_classes, (1, 1), padding='same', activation='softmax')(x)

    # Create model
    model = Model(inputs, outputs)

    return model


def custom_vgg19( input_shape, classes ):
    model = tf.keras.applications.VGG19 (weights="imagenet", include_top=False, input_shape=input_shape, classes=classes)
    x = model.layers[-1].output
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Conv2DTranspose(512, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(256, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(128, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    x = Conv2DTranspose(64, (2, 2), strides=(2, 2), activation='relu', padding='same')(x)
    conv9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), activation='softmax', padding='same')(x)
    outputs = Conv2D(classes, 1, padding='same')( conv9 )
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
    out_24 = convolution_block(dspp_input, kernel_size=3, dilation_rate=24)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_6, out_12, out_18, out_24])
    output = convolution_block(x, kernel_size=1)
    return output

def DilatedSpatialPyramidPoolingD4(dspp_input):
    dims = dspp_input.shape
    x = AveragePooling2D(pool_size=(dims[-3], dims[-2]))(dspp_input)
    x = convolution_block(x, kernel_size=1, use_bias=True)
    out_pool = layers.UpSampling2D(
        size=(dims[-3] // x.shape[1], dims[-2] // x.shape[2]), interpolation="bilinear",
    )(x)

    out_1 = convolution_block(dspp_input, kernel_size=1, dilation_rate=1)
    out_4 = convolution_block(dspp_input, kernel_size=3, dilation_rate=4)
    out_8 = convolution_block(dspp_input, kernel_size=3, dilation_rate=8)
    out_12 = convolution_block(dspp_input, kernel_size=3, dilation_rate=12)
    out_16 = convolution_block(dspp_input, kernel_size=3, dilation_rate=16)
    out_20 = convolution_block(dspp_input, kernel_size=3, dilation_rate=20)
    out_24 = convolution_block(dspp_input, kernel_size=3, dilation_rate=24)

    x = layers.Concatenate(axis=-1)([out_pool, out_1, out_4, out_8, out_12, out_16, out_20, out_24])
    output = convolution_block(x, kernel_size=1)
    return output

def DeeplabV3Plus(image_size, num_classes, output_activation='softmax',is_pretrained=True):
    model_input = Input(shape=(image_size[0], image_size[1], image_size[2]))
    weights=None

    if is_pretrained:
        weights="imagenet"

    resnet50 = applications.ResNet50(
        weights=weights, include_top=False, input_tensor=model_input
    )

    # weights = "imagenet", include_top = False, input_tensor = model_input

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

def DeeplabV3PlusD4(image_size, num_classes, output_activation='softmax',is_pretrained=True):
    model_input = Input(shape=(image_size[0], image_size[1], image_size[2]))
    weights=None

    if is_pretrained:
        weights="imagenet"

    resnet50 = applications.ResNet50(
        weights=weights, include_top=False, input_tensor=model_input
    )

    # weights = "imagenet", include_top = False, input_tensor = model_input

    x = resnet50.get_layer("conv4_block6_2_relu").output
    x = DilatedSpatialPyramidPoolingD4(x)

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

def DeeplabV3Plus101(image_size, num_classes, output_activation='softmax'):
    model_input = Input(shape=(image_size[0], image_size[1], image_size[2]))
    resnet50 = applications.ResNet101(
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