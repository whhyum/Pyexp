# import tensorflow as tf
# import numpy as np
# import random
#
#
# def int64_feature(value):
#     """Wrapper for inserting int64 features into Example proto."""
#     if not isinstance(value, list):
#         value = [value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
#
#
# def bytes_feature(value):
#     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
#
#
# def imgLabel2example(img_path, classes):
#     with tf.io.gfile.GFile(img_path, 'rb') as fid:
#         encoded_jpg = fid.read()
#     example = tf.train.Example(features=tf.train.Features(feature={
#         'image/filename': bytes_feature(img_path.encode('utf8')),
#         'image/encoded': bytes_feature(encoded_jpg),
#         'image/class': int64_feature(int(classes)),
#     }))
#
#     return example
#
#
# trainRecordPath = "./train.record"  # 训练集的record数据存储的位置
# testRecordPath = "./test.record"  # 测试集的record数据存储的位置
# writerTrain = tf.io.TFRecordWriter(trainRecordPath)  # 打开并写入
# writerTest = tf.io.TFRecordWriter(testRecordPath)
#
# txt_path = "./DogVsCat/train.txt"  # 存放图像位置和对应label的txt文件
# data_lib = open(txt_path, "r").readlines()
# random.shuffle(data_lib)
#
# trainLib = data_lib[: int(len(data_lib) * 0.9)]
# testLib = data_lib[int(len(data_lib) * 0.9):]
#
# for line in testLib:
#     img_path = line.strip().split(" ")[0]
#     label = int(line.strip().split(" ")[1])
#     # 生成tf.train.feature属性(key和value的键值对)，在将这些单独feature整合成features
#     # 生成Example并序列化
#     example = imgLabel2example(img_path, label)
#     # 将example序列化，压缩以减少size
#     serialized_example = example.SerializeToString()
#     writerTest.write(serialized_example)
# writerTest.close()
#
# for line in trainLib:
#     img_path = line.strip().split(" ")[0]
#     label = int(line.strip().split(" ")[1])
#     # 生成tf.train.feature属性(key和value的键值对)，在将这些单独feature整合成features
#     # 生成Example并序列化
#     examples = imgLabel2example(img_path, label)
#     # 将example序列化，压缩以减少size
#     serialized_example = examples.SerializeToString()
#     writerTrain.write(serialized_example)
#
# writerTrain.close()
# print("sess write")
