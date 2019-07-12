from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

'''
变量表示共享持久状态
可以在多个sess.run 调用的上下文之外，多个sess 之间可见
'''

'''
创建变量
'''
my_variable = tf.get_variable("my_variable",[1,2,3])

