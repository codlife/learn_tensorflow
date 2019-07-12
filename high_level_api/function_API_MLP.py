import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
print(tf.VERSION)
print(tf.keras.__version__)

'''
tf.keras.Sequential模型是层的简单堆叠，无法表示任意模型
使用keras函数式API可以构建更加复杂的模型拓扑，例如
多输入模型
多输出模型
具有共享层的模型（同一层被调用多次）
具有非序列数据流的模型（例如：剩余连接）

函数式API特点
1：层实例可调用并返回张量
2：输入张量和输出张量用于定义tf.keras.Model实例
3：模型训练和Sequential一致
'''
# a layer instance is callable on a tensor and return a tensor
inputs = tf.keras.Input(shape = (32,))
x = layers.Dense(64, activation = 'relu')(inputs)
x = layers.Dense(64, activation = 'relu')(x)
predictions = layers.Dense(10, activation = 'softmax')(x)

model = tf.keras.Model(inputs = inputs, outputs = predictions)
model.compile(optimizer = tf.train.RMSPropOptimizer(0.001),
        loss = 'categorical_crossentropy',
        metrics = ['accuracy'])

data = np.random.random((1000,32))
labels = np.random.random((1000,10))

model.fit(data, labels, batch_size = 32, epochs = 5)

