from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras import Model
from tensorflow.keras.applications.resnet50 import preprocess_input
import tensorflow as tf

from load_data import Load_Data

"""加载数据"""
train_ds, val_ds, class_names = Load_Data(data_dir='dataset/outputs', batch_size=32, input_height=180, input_width=180)

"""网络"""
# 定义model
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(180, 180, 3))
base_model.trainable = False
inputs = tf.keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
print(base_model.output.shape)
x = GlobalAveragePooling2D()(x)
y = Dense(2, activation='softmax')(x)  # final layer with softmax activation
model = Model(inputs=inputs, outputs=y, name="ResNet50")
model.summary()

"""训练"""
# 编译模型
loss = tf.keras.losses.SparseCategoricalCrossentropy()
metrics = tf.metrics.SparseCategoricalAccuracy()
model.compile(optimizer='Adam', loss=loss, metrics=metrics)
len(model.trainable_variables)

# 训练模型
history = model.fit(train_ds,
                    epochs=10,
                    validation_data=val_ds)
# 保存模型
model.save("model.h5")


import matplotlib.pyplot as plt

# 绘制训练集的准确度和损失曲线
plt.figure(figsize=(12, 4))

# 绘制训练集准确度曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['sparse_categorical_accuracy'])
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

# 绘制训练集损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.title('Training Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()

# 绘制验证集的准确度和损失曲线
plt.figure(figsize=(12, 4))

# 绘制验证集准确度曲线
plt.subplot(1, 2, 1)
plt.plot(history.history['val_sparse_categorical_accuracy'])
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

# 绘制验证集损失曲线
plt.subplot(1, 2, 2)
plt.plot(history.history['val_loss'])
plt.title('Validation Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.tight_layout()
plt.show()
