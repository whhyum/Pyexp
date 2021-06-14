import tensorflow as tf
from tqdm import tqdm
learning_rate = 0.001
training_step = 30000

batch_size = 16
train_record_path = "./dataset/train.record"
test_record_path = "./dataset/test.record"
# 调用后我们会得到一个Dataset(tf.data.Dataset)，字面理解，这里面就存放着我们之前写入的所有Example。
train_dataset = tf.data.TFRecordDataset(train_record_path)
test_dataset = tf.data.TFRecordDataset(test_record_path)
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
    image = tf.image.resize_with_crop_or_pad(image, 224, 224)
    image = tf.reshape(image, [224, 224, 3])
    image = tf.cast(image, tf.float32)

    feature_dict['image'] = image
    return feature_dict['image'], feature_dict['image/class']
#
#
train_dataset = train_dataset.map(parese_example)
test_dataset = test_dataset.map(parese_example)

train_dataset = train_dataset.repeat().shuffle(2000).batch(batch_size).prefetch(3)
test_dataset = test_dataset.repeat().shuffle(2000).batch(batch_size).prefetch(3)

# model = vgg16.vgg16()
ResNet50 = tf.keras.applications.ResNet50(weights=None, include_top=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
fc = tf.keras.layers.Dense(2, activation="softmax") # 修改乘自己的类别数
model = tf.keras.Sequential([ResNet50, global_average_layer, fc])


model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=["accuracy"])

print(model.summary())
model.fit(train_dataset, epochs=5, validation_data=test_dataset,  shuffle=True, steps_per_epoch=1000, validation_steps=1)
model.save("./resnet.h5")
