import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")
