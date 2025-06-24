import tensorflow_hub as hub

import keras_cv
from tensorflow.keras import layers, models


def create_vit_model(input_shape, output_shape)
    inputs = layers.Input(shape=input_shape)

    # Use KerasCV pretrained ViT encoder
    vit_encoder = keras_cv.models.VisionTransformer(
        include_top=False,
        input_shape=input_shape,
        patch_size=16,
        num_layers=12,
        hidden_dim=768,
        num_heads=12,
        mlp_dim=3072,
        dropout=0.1,
        classifier_activation=None,
        weights="imagenet21k",  # or "imagenet2012"
    )

    x = vit_encoder(inputs)  # shape: (batch_size, 10*20, 768) (patch embeddings)

    # Reshape patch embeddings back to grid (num_patches_y, num_patches_x, hidden_dim)
    num_patches_y = input_shape[0] // 16
    num_patches_x = input_shape[1] // 16
    x = layers.Reshape((num_patches_y, num_patches_x, 768))(x)

    # Decoder: simple Conv2DTranspose upsampling
    x = layers.Conv2DTranspose(256, 3, strides=2, padding="same", activation="relu")(x)  # 10x20 -> 20x40
    x = layers.Conv2DTranspose(128, 3, strides=2, padding="same", activation="relu")(x)  # 20x40 -> 40x80
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu")(x)   # 40x80 -> 80x160
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)   # 80x160 -> 160x320

    outputs = layers.Conv2D(num_classes, 1, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    model.summary()
