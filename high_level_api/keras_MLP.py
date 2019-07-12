import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

#定义模型
model = tf.keras.Sequential()
model.add(layers.Dense(64, activation = 'relu'))
model.add(layers.Dense(64, activation = 'relu'))

model.add(layers.Dense(10, activation = 'softmax'))

#训练评估
model.compile(optimizer = tf.train.AdamOptimizer(0.001),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'],
        )

#输入numpy 数据
import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit(data, labels, epochs=10, batch_size=32)

#使用Dataset API 可扩展为大型数据集或多设备训练
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()
#model.fit(dataset, epochs = 10, steps_per_epoch = 30)

val_data = np.random.random((100,32))
val_labels = np.random.random((100,10))

#model.fit(data, labels, epochs = 10, batch_size = 32,
#        validation_data = (val_data, val_labels))
#使用tf.Dataset API 可以扩展为大型数据集或多设备训练，

dataset = tf.data.Dataset.from_tensor_slices((data,labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()
model.fit(dataset, epochs = 10, steps_per_epoch=30)


data = np.random.random((1000, 32))
labels = np.random.random((1000,10))
dataset = tf.data.Dataset.from_tensor_slices((data, labels))
dataset = dataset.batch(32)
dataset = dataset.repeat()
print(model.evaluate(data, labels, batch_size = 32))
model.evaluate(dataset, steps = 30)
result = model.predict(data, batch_size=32)
#print(result)
#print(result - labels)
print(result.shape)
