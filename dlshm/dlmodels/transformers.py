import tensorflow_hub as hub

import keras_cv
from tensorflow.keras import layers, models
from transformers import ViTModel
import tensorflow as tf
from transformers import TFAutoModel

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        input_shape = ops.shape(images)
        batch_size = input_shape[0]
        height = input_shape[1]
        width = input_shape[2]
        channels = input_shape[3]
        num_patches_h = height // self.patch_size
        num_patches_w = width // self.patch_size
        patches = tf.keras.ops.image.extract_patches(images, size=self.patch_size)
        patches = tf.keras.ops.reshape(
            patches,
            (
                batch_size,
                num_patches_h * num_patches_w,
                self.patch_size * self.patch_size * channels,
            ),
        )
        return patches

    def get_config(self):
        config = super().get_config()
        config.update({"patch_size": self.patch_size})
        return config

def create_vit_classifier(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    # # Augment data.
    # augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(inputs)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(num_classes)(features)
    # Create the Keras model.
    model = keras.Model(inputs=inputs, outputs=logits)
    return model

def create_vit_model(input_shape, num_classes ):
    inputs = layers.Input(shape=input_shape)



    vit = TFAutoModel.from_pretrained("google/vit-base-patch16-224-in21k")

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
