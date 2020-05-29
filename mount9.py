
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
tn.set_default_backend("tensorflo

                      
  def __init__(self):
    super(TNLayer, self).__init__()
    self.a_var = tf.Variable(tf.random.normal(
            shape=(8, 8, 2), stddev=1.0/16.0),
             name="a", trainable=True)
    self.b_var = tf.Variable(tf.random.normal(shape=(8, 8, 2), stddev=1.0/16.0),
                             name="b", trainable=True)
    self.bias = tf.Variable(tf.zeros(shape=(8, 8)), name="bias", trainable=True)

                       
                       
  #def call(self, inputs):
