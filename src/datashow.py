import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

record_path = "./dataset/train.record"

# 调用后我们会得到一个Dataset(tf.data.Dataset)，字面理解，这里面就存放着我们之前写入的所有Example。
dataset = tf.data.TFRecordDataset(record_path)
# 定义一个解析函数

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
#     print (np.frombuffer(row['image/class'].numpy(), dtype=np.uint8)) # 如果要恢复成3d数组，可reshape
