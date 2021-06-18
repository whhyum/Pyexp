import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from keras import layers, optimizers, models
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import load_model
from keras.models import Model

learning_rate = 0.0001
# training_step = 30000

epochs = 200
batch_size = 32

train_record_path = "./dataset/train.record"
test_record_path = "./dataset/test.record"
# 调用后我们会得到一个Dataset(tf.data.Dataset),这里面就存放着写入的所有Example。
train_dataset = tf.data.TFRecordDataset(train_record_path)
test_dataset = tf.data.TFRecordDataset(test_record_path)

steps_per_epoch = 30
    # 25000/batch_size
validation_steps = 10
    # 12500/batch_size


# 定义一个解析函数

feature_description = {
    'image/filename': tf.io.FixedLenFeature([], tf.string),
    'image/class': tf.io.FixedLenFeature([], tf.int64),
    'image/encoded': tf.io.FixedLenFeature([], tf.string)
}

#
def parese_example(serialized_example):
    feature_dict = tf.io.parse_single_example(serialized_example, feature_description)
    image = tf.io.decode_jpeg(feature_dict['image/encoded'])  # 解码JPEG图片
    shape1 = [128, 128]
    image = tf.image.resize(image, shape1)
    image = tf.reshape(image, [128, 128, 3])
    image = tf.cast(image, tf.float32)

    feature_dict['image'] = image
    return feature_dict['image'], feature_dict['image/class']
#
#
train_dataset = train_dataset.map(parese_example)
test_dataset = test_dataset.map(parese_example)

train_dataset = train_dataset.repeat().shuffle(2000).batch(batch_size).prefetch(3)
test_dataset = test_dataset.repeat().shuffle(2000).batch(batch_size).prefetch(3)

path = '../source/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'
# model = vgg16.vgg16()
# 预训练模型1
# path = "../source/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5"
ResNet = tf.keras.applications.ResNet50(weights= path, include_top=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
fc = tf.keras.layers.Dense(2, activation="softmax") # 修改乘自己的类别数
model = tf.keras.Sequential([ResNet, global_average_layer, fc])

# ResNet50.trainable =False
# #模型搭建
# model = tf.keras.Sequential()
# model.add(ResNet50)
# model.add(tf.keras.layers.GlobalAveragePooling2D())
# model.add(tf.keras.layers.Dense(512,activation='relu'))
# model.add(tf.keras.layers.Dense(1,activation='sigmoid'))

# 预训练模型2

# conv_base  = tf.keras.applications.ResNet50(weights=path, include_top=False, input_shape=(150, 150, 3))
#
# model = models.Sequential()
# model.add(conv_base)
# model.add(layers.Flatten())
# model.add(layers.Dense(2, activation='sigmoid'))
#
# conv_base.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              # loss='binary_crossentropy',
              # loss= 'mean_squared_error',
              metrics=["accuracy"])

print(model.summary())
model_train = model.fit(train_dataset, epochs = epochs, validation_data=test_dataset, shuffle=True, steps_per_epoch = steps_per_epoch, validation_steps=validation_steps)
# model.save("./debug2_resnet.h5")
# save weights only
model.save_weights('./model/weights6.19.h5')

#训练结果可视化
accuracy = model_train.history["accuracy"]
test_accuracy = model_train.history["val_accuracy"]
loss = model_train.history["loss"]
test_loss = model_train.history["val_loss"]
epochs_range = range(epochs)
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.plot(epochs_range,accuracy,label = "Training Acc")
plt.plot(epochs_range,test_accuracy,label = "Test Acc")
plt.legend()
plt.title("Training and Test Acc")

plt.subplot(1,2,2)
plt.plot(epochs_range,loss,label = "Training loss")
plt.plot(epochs_range,test_loss,label = "Test loss")
plt.legend()
plt.title("Training and Test loss")
plt.show()
