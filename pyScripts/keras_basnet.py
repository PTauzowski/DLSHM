# Highly accurate boundaries segmentation using BASNet
#
# Author: Hamid Ali
# Date created: 2023/05/30
# Last modified: 2025/01/24
# Description: Boundaries aware segmentation model trained on the DUTS dataset.
# Taken from Keras examples


import os

# Because of the use of tf.image.ssim in the loss,
# this example requires TensorFlow. The rest of the code
# is backend-agnostic.
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
from glob import glob
import matplotlib.pyplot as plt

import keras_hub
import tensorflow as tf
import keras
from keras import layers, ops

keras.config.disable_traceback_filtering()


IMAGE_SIZE = 288
BATCH_SIZE = 4
OUT_CLASSES = 1
TRAIN_SPLIT_RATIO = 0.90


# data_dir = keras.utils.get_file(
#     origin="http://saliencydetection.net/duts/download/DUTS-TE.zip",
#     extract=True,
# )
data_dir = '/Users/piotrek/Computations/Ai/Data/DUTS-TE'


def load_paths(path, split_ratio):
    images = sorted(glob(os.path.join(path, "DUTS-TE-Image/*")))[:140]
    masks = sorted(glob(os.path.join(path, "DUTS-TE-Mask/*")))[:140]
    len_ = int(len(images) * split_ratio)
    return (images[:len_], masks[:len_]), (images[len_:], masks[len_:])


class Dataset(keras.utils.PyDataset):
    def __init__(
        self,
        image_paths,
        mask_paths,
        img_size,
        out_classes,
        batch,
        shuffle=True,
        **kwargs,
    ):
        if shuffle:
            perm = np.random.permutation(len(image_paths))
            image_paths = [image_paths[i] for i in perm]
            mask_paths = [mask_paths[i] for i in perm]
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_size = img_size
        self.out_classes = out_classes
        self.batch_size = batch
        super().__init__(*kwargs)

    def __len__(self):
        return len(self.image_paths) // self.batch_size

    def __getitem__(self, idx):
        batch_x, batch_y = [], []
        for i in range(idx * self.batch_size, (idx + 1) * self.batch_size):
            x, y = self.preprocess(
                self.image_paths[i],
                self.mask_paths[i],
                self.img_size,
            )
            batch_x.append(x)
            batch_y.append(y)
        batch_x = np.stack(batch_x, axis=0)
        batch_y = np.stack(batch_y, axis=0)
        return batch_x, batch_y

    def read_image(self, path, size, mode):
        x = keras.utils.load_img(path, target_size=size, color_mode=mode)
        x = keras.utils.img_to_array(x)
        x = (x / 255.0).astype(np.float32)
        return x

    def preprocess(self, x_batch, y_batch, img_size):
        images = self.read_image(x_batch, (img_size, img_size), mode="rgb")  # image
        masks = self.read_image(y_batch, (img_size, img_size), mode="grayscale")  # mask
        return images, masks


train_paths, val_paths = load_paths(data_dir, TRAIN_SPLIT_RATIO)

train_dataset = Dataset(
    train_paths[0], train_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=True
)
val_dataset = Dataset(
    val_paths[0], val_paths[1], IMAGE_SIZE, OUT_CLASSES, BATCH_SIZE, shuffle=False
)

def display(display_list):
    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(keras.utils.array_to_img(display_list[i]), cmap="gray")
        plt.axis("off")
    plt.show()


for image, mask in val_dataset:
    display([image[0], mask[0]])
    break

print(f"Unique values count: {len(np.unique((mask[0] * 255)))}")
print("Unique values:")
print(np.unique((mask[0] * 255)).astype(int))



def basic_block(x_input, filters, stride=1, down_sample=None, activation=None):
    """Creates a residual(identity) block with two 3*3 convolutions."""
    residual = x_input

    x = layers.Conv2D(filters, (3, 3), strides=stride, padding="same", use_bias=False)(
        x_input
    )
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(filters, (3, 3), strides=(1, 1), padding="same", use_bias=False)(
        x
    )
    x = layers.BatchNormalization()(x)

    if down_sample is not None:
        residual = down_sample

    x = layers.Add()([x, residual])

    if activation is not None:
        x = layers.Activation(activation)(x)

    return x


def convolution_block(x_input, filters, dilation=1):
    """Apply convolution + batch normalization + relu layer."""
    x = layers.Conv2D(filters, (3, 3), padding="same", dilation_rate=dilation)(x_input)
    x = layers.BatchNormalization()(x)
    return layers.Activation("relu")(x)


def segmentation_head(x_input, out_classes, final_size):
    """Map each decoder stage output to model output classes."""
    x = layers.Conv2D(out_classes, kernel_size=(3, 3), padding="same")(x_input)

    if final_size is not None:
        x = layers.Resizing(final_size[0], final_size[1])(x)

    return x


def get_resnet_block(resnet, block_num):
    """Extract and return a ResNet-34 block."""
    extractor_levels = ["P2", "P3", "P4", "P5"]
    num_blocks = resnet.stackwise_num_blocks
    if block_num == 0:
        x = resnet.get_layer("pool1_pool").output
    else:
        x = resnet.pyramid_outputs[extractor_levels[block_num - 1]]
    y = resnet.get_layer(f"stack{block_num}_block{num_blocks[block_num]-1}_add").output
    return keras.models.Model(
        inputs=x,
        outputs=y,
        name=f"resnet_block{block_num + 1}",
    )



def basnet_predict(input_shape, out_classes):
    """BASNet Prediction Module, it outputs coarse label map."""
    filters = 64
    num_stages = 6

    x_input = layers.Input(input_shape)

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    resnet = keras_hub.models.ResNetBackbone(
        input_conv_filters=[64],
        input_conv_kernel_sizes=[7],
        stackwise_num_filters=[64, 128, 256, 512],
        stackwise_num_blocks=[3, 4, 6, 3],
        stackwise_num_strides=[1, 2, 2, 2],
        block_type="basic_block",
    )

    encoder_blocks = []
    for i in range(num_stages):
        if i < 4:  # First four stages are adopted from ResNet-34 blocks.
            x = get_resnet_block(resnet, i)(x)
            encoder_blocks.append(x)
            x = layers.Activation("relu")(x)
        else:  # Last 2 stages consist of three basic resnet blocks.
            x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            x = basic_block(x, filters=filters * 8, activation="relu")
            encoder_blocks.append(x)

    # -------------Bridge-------------
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    x = convolution_block(x, filters=filters * 8, dilation=2)
    encoder_blocks.append(x)

    # -------------Decoder-------------
    decoder_blocks = []
    for i in reversed(range(num_stages)):
        if i != (num_stages - 1):  # Except first, scale other decoder stages.
            shape = x.shape
            x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)

        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        x = convolution_block(x, filters=filters * 8)
        decoder_blocks.append(x)

    decoder_blocks.reverse()  # Change order from last to first decoder stage.
    decoder_blocks.append(encoder_blocks[-1])  # Copy bridge to decoder.

    # -------------Side Outputs--------------
    decoder_blocks = [
        segmentation_head(decoder_block, out_classes, input_shape[:2])
        for decoder_block in decoder_blocks
    ]

    return keras.models.Model(inputs=x_input, outputs=decoder_blocks)

def basnet_rrm(base_model, out_classes):
    """BASNet Residual Refinement Module(RRM) module, output fine label map."""
    num_stages = 4
    filters = 64

    x_input = base_model.output[0]

    # -------------Encoder--------------
    x = layers.Conv2D(filters, kernel_size=(3, 3), padding="same")(x_input)

    encoder_blocks = []
    for _ in range(num_stages):
        x = convolution_block(x, filters=filters)
        encoder_blocks.append(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(x)

    # -------------Bridge--------------
    x = convolution_block(x, filters=filters)

    # -------------Decoder--------------
    for i in reversed(range(num_stages)):
        shape = x.shape
        x = layers.Resizing(shape[1] * 2, shape[2] * 2)(x)
        x = layers.concatenate([encoder_blocks[i], x], axis=-1)
        x = convolution_block(x, filters=filters)

    x = segmentation_head(x, out_classes, None)  # Segmentation head.

    # ------------- refined = coarse + residual
    x = layers.Add()([x_input, x])  # Add prediction + refinement output

    return keras.models.Model(inputs=[base_model.input], outputs=[x])


class BASNet(keras.Model):
    def __init__(self, input_shape, out_classes):
        """BASNet, it's a combination of two modules
        Prediction Module and Residual Refinement Module(RRM)."""

        # Prediction model.
        predict_model = basnet_predict(input_shape, out_classes)
        # Refinement model.
        refine_model = basnet_rrm(predict_model, out_classes)

        output = refine_model.outputs  # Combine outputs.
        output.extend(predict_model.output)

        # Activations.
        output = [layers.Activation("sigmoid")(x) for x in output]
        super().__init__(inputs=predict_model.input, outputs=output)

        self.smooth = 1.0e-9
        # Binary Cross Entropy loss.
        self.cross_entropy_loss = keras.losses.BinaryCrossentropy()
        # Structural Similarity Index value.
        self.ssim_value = tf.image.ssim
        # Jaccard / IoU loss.
        self.iou_value = self.calculate_iou

    def calculate_iou(
        self,
        y_true,
        y_pred,
    ):
        """Calculate intersection over union (IoU) between images."""
        intersection = ops.sum(ops.abs(y_true * y_pred), axis=[1, 2, 3])
        union = ops.sum(y_true, [1, 2, 3]) + ops.sum(y_pred, [1, 2, 3])
        union = union - intersection
        return ops.mean((intersection + self.smooth) / (union + self.smooth), axis=0)

    def compute_loss(self, x, y_true, y_pred, sample_weight=None, training=False):
        total = 0.0
        for y_pred_i in y_pred:  # y_pred = refine_model.outputs + predict_model.output
            cross_entropy_loss = self.cross_entropy_loss(y_true, y_pred_i)

            ssim_value = self.ssim_value(y_true, y_pred, max_val=1)
            ssim_loss = ops.mean(1 - ssim_value + self.smooth, axis=0)

            iou_value = self.iou_value(y_true, y_pred)
            iou_loss = 1 - iou_value

            # Add all three losses.
            total += cross_entropy_loss + ssim_loss + iou_loss
        return total



basnet_model = BASNet(
    input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], out_classes=OUT_CLASSES
)  # Create model.
basnet_model.summary()  # Show model summary.

optimizer = keras.optimizers.Adam(learning_rate=1e-4, epsilon=1e-8)
# Compile model.
basnet_model.compile(
    optimizer=optimizer,
    metrics=[keras.metrics.MeanAbsoluteError(name="mae") for _ in basnet_model.outputs],
)

basnet_model.fit(train_dataset, validation_data=val_dataset, epochs=1)

def normalize_output(prediction):
    max_value = np.max(prediction)
    min_value = np.min(prediction)
    return (prediction - min_value) / (max_value - min_value)

for (image, mask), _ in zip(val_dataset, range(1)):
    pred_mask = basnet_model.predict(image)
    display([image[0], mask[0], normalize_output(pred_mask[0][0])])