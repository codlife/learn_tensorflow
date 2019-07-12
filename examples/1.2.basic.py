# -*- coding: utf-8 -*-
'''
使用图graph来表示计算任务
在被称之为session的上下文context中执行任务
使用tensor表示数据
通过变量维护状态
使用feed和fetch可以为任意的操作赋值或者从其中获取数据
'''
'''
综述
构建图
tf程序有一个默认图，op构造器可以为其增加节点
'''
import tensorflow as tf
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
#构造完成之后，需要启动一个会话
with tf.Session() as sess:
	with tf.device("/cpu:0"):	
		result = sess.run(product)
		print(result)

print("wjf")

