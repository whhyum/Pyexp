import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

test_record_path = "./dataset/test.record"

dataset = tf.data.TFRecordDataset(test_record_path)

feature_description = {
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/class': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}


def parese_example(serialized_example):
    feature_dict = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image/encoded'])  # 解码JPEG图片
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.cast(image, tf.float32)

    feature_dict['image'] = image
    return feature_dict['image'], feature_dict['image/class']


dataset = dataset.map(parese_example)
dataset = dataset.repeat().shuffle(5000).batch(1).prefetch(1)

for img, label in dataset.take(1):  # 只取前1条
    print(label.numpy())
    print(img.numpy()[0].shape)
    plt.imshow(np.array(img.numpy()[0], np.uint8))
    plt.show()