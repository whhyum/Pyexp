import tensorflow as tf
import os
total_num = 25000
#参数设置
learning_rate = 0.001
test_step = 1000
saved_step = 5000
EPOCHS = 10

batch_size = 16

display_step = 10

training_step = int(total_num / batch_size)

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

train_dataset = train_dataset.repeat().shuffle(5000).batch(batch_size).prefetch(3)
test_dataset = test_dataset.repeat().shuffle(5000).batch(batch_size, drop_remainder=True)

ResNet50 = tf.keras.applications.ResNet50(weights=None, include_top=False)
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
fc = tf.keras.layers.Dense(2, activation="softmax")
model = tf.keras.Sequential([ResNet50, global_average_layer, fc])

# # Choose an optimizer and loss function for training
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

# # Select metrics to measure the loss and the accuracy of the model
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

optimizer = tf.keras.optimizers.Adam(0.001)
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
       # print("=> label shape: ", labels.shape, "pred shape", predictions.shape)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)
print("train..")

def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)

for epoch in range(EPOCHS):
    for step, (batch_x, batch_y) in enumerate(train_dataset, 1):
        train_step(batch_x, batch_y)
        if(step % display_step == 0):
          template = '=> train: step {}, Loss: {:.4}, Accuracy: {:.2%}'
          print(template.format(step+1,
                                train_loss.result(),
                                 train_accuracy.result(),
                                ))
    for step, (batch_x, batch_y) in enumerate(test_dataset, 1):
        test_step(batch_x, batch_y)

    template = '=> Epoch {}, , Test Loss: {:.4}, Test Accuracy: {:.2%}'
    print(template.format(epoch + 1,
                          test_loss.result(),
                          test_accuracy.result()))

    root = tf.train.Checkpoint(optimizer=optimizer,
                                model=model)
    saved_folder = "./ckpt2Model"
    if(not os.path.exists(saved_folder)):
        os.mkdir(saved_folder)
    checkpoint_prefix = (saved_folder + "/epoch:%i_acc") % (epoch + 1)
    root.save(checkpoint_prefix)
