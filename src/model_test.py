import tensorflow as tf
module_path = "../source/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
ResNet18 = tf.keras.applications.ResNet50(weights=module_path, include_top=False)
print(ResNet18.summary())
x = tf.random.normal([8,224,224,3])
out = ResNet18(x)
print(out.shape)
