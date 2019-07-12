#! /bin/sh
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

'''
tensorflow core 主要包括两部分
1 图的构建 tf.Graph
2 图的计算 tf.Session
'''
a = tf.constant(3.0, dtype = tf.float32)
b = tf.constant(4.0)
total = a + b
print(a)
print(b)
print(total)

with tf.Session() as sess:
	print(sess.run(total))
	print(sess.run({'ab':(a,b), 'total': total}))

sess = tf.Session()
vec = tf.random_uniform(shape=(3,))
out1 = vec + 1
out2 = vec + 2
print(sess.run(vec))
print(sess.run(vec))
print(sess.run((out1, out2)))

'''
供给 placeholder
占位符必须在后面给出值，否则会报错
可以认为是函数参数
'''
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
z = x + y
#使用run 方法的feed_dict 参数为占位符提供具体的值
print(sess.run(z, feed_dict = {x: 3, y: 4.5}))
print(sess.run(z, feed_dict = {x: [1,3], y: [2, 4]}))

'''
数据集
占位符适合简单的实验，数据集是将数据流试传输到模型的首选
需转化成可以使用的tf.Tensor,需要先转化成 tf.data.Iterator
然后调用迭代器的get_next
创建迭代器最简单的方式是：make_one_shot_iterator
'''

my_data = [
	[0, 1],
	[2, 3],
	[4, 5],
]
slices = tf.data.Dataset.from_tensor_slices(my_data)
next_item = slices.make_one_shot_iterator().get_next()
while True:
	try:
		print(sess.run(next_item))
	except tf.errors.OutOfRangeError:
		break

'''
layer
创建层
'''
x = tf.placeholder(tf.float32,shape=[None, 3])

linear_model = tf.layers.Dense(units = 1)
y = linear_model(x)

'''
初始层
'''
init = tf.global_variables_initializer()
sess.run(init)

'''
执行层
'''
print(sess.run(y, {x:[[1,2,3],[4,5,5]]}))



