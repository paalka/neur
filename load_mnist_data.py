import struct
import numpy as np

"""
MNIST image input image format:

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel

Pixels are organized row-wise.
Pixel values are grayscale 0-255.

Source: http://yann.lecun.com/exdb/mnist/
"""

IMG_MAGIC_NUM = 2051
def parse_mnist_imgs(path_to_imgs, expected_magic_number=IMG_MAGIC_NUM):
    with open(path_to_imgs, 'rb') as binary_img_file:
        # Obtain info from the header
        read_magic_num, n_images, n_rows, n_cols = struct.unpack(">IIII", binary_img_file.read(16))

        if read_magic_num != expected_magic_number:
            print("Magic num mismatch! Expected: {} Got: {}".format(expected_magic_number, read_magic_num))

        images = np.fromfile(binary_img_file, dtype=np.uint8).reshape(n_images, n_rows * n_cols)

        return images

"""
MNIST label input image format:

    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  10000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label

Where each label is a number between 1-8

Source: http://yann.lecun.com/exdb/mnist/
"""

LABELS_MAGIC_NUM = 2049
def parse_mnist_labels(path_to_labels, expected_magic_number=LABELS_MAGIC_NUM):
    with open(path_to_labels, 'rb') as binary_labels_file:
        read_magic_num, n_labes = struct.unpack(">II", binary_labels_file.read(8))

        if read_magic_num != expected_magic_number:
            print("Magic num mismatch! Expected: {} Got: {}".format(expected_magic_number, read_magic_num))

        labels = np.fromfile(binary_labels_file, dtype=np.uint8)

        return labels
