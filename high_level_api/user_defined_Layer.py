import tensorflow as tf
from tensorflow.keras import layers

print(tf.VERSION)
print(tf.keras.__version__)

'''
继承tf.keras.layers.Layer 自定义层
build: 创建层的权重，使用add_weight 方法添加权重
call：定义前向传播
compute_output_shape: 制定在给定输入情况下如何计算层的输出形状

'''
# 自定义一个通过核矩阵实现输入matmul的自定义层示例
class MyLayer(layers.Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(MyLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1], self.output_dim))
        
        self.kernel = self.add_weight(name = 'kernel',
                shape = shape,
                initializer = 'uniform',
                trainable = True)
        #be sure to call this at the end
        super(MyLayer, self).build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)

    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.output_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer, self).get_config()
        base_config['output_dim'] = self.output_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


model = tf.keras.Sequential([MyLayer(10),
    layers.Activation('softmax')])

# The compile step specifies the training configuration.
model.compile(optimizer=tf.train.RMSPropOptimizer(0.001),
                  loss='categorical_crossentropy',
                                metrics=['accuracy'])
# Trains for 5 epochs.
import numpy as np
data = np.random.random((1000, 32))
labels = np.random.random((1000, 10))
model.fit(data, labels, batch_size=32, epochs=5)

