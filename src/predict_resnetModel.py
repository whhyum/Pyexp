import tensorflow as tf
import numpy as np
import cv2
cv2.__version__

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
#
#
# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
#

def build_model():
    # based on VGG-16
    preModel_path = "../source/resnet50_weights_tf_dim_ordering_tf_kernels_notop (1).h5"
    ResNet18 = tf.keras.applications.ResNet50(weights=preModel_path, include_top=False)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    fc = tf.keras.layers.Dense(2, activation="softmax")  # 修改乘自己的类别数
    model = tf.keras.Sequential([ResNet18, global_average_layer, fc])
    # configure the model
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    model.summary()
    return model

pretrained_model = build_model()
pretrained_model.load_weights('./weights.h5')

# model_path = "./resnet.h5"
# model = tf.keras.models.load_model(model_path)
img = cv2.imread("../data/test/644.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (64, 64))
img = np.array(img, np.float32)
x = np.expand_dims(img, 0)
y = pretrained_model.predict(x)
predict_cat = y[:, 0]
predict_dog = y[:, 1]
print("predict: ", y)
print("猫的概率: ", predict_cat * 100)
print("狗的概率: ", predict_dog * 100)
