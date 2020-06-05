
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
tf.enable_v2_behavior()
import tensornetwork as tn
tn.set_default_backend("tensorflow")
                      
  def __init__(self):
    super(TNLayer, self).__init__()
    self.a_var = tf.Variable(tf.random.normal(
            shape=(8, 8, 2), stddev=1.0/16.0),
             name="a", trainable=True)
    self.b_var = tf.Variable(tf.random.normal(shape=(8, 8, 2), stddev=1.0/16.0),
                             name="b", trainable=True)
    self.bias = tf.Variable(tf.zeros(shape=(8, 8)), name="bias", trainable=True)
                
                       
  def call(self, inputs):
    def f(input_vec, a_var, b_var, bias_var):
      input_vec = tf.reshape(input_vec, (8,8))
      a = tn.Node(a_var)
      b = tn.Node(b_var)
                    
      x_node = tn.Node(input_vec)
      a[1] ^ x_node[0]
      b[1] ^ x_node[1]
      a[2] ^ b[2]

      c = a @ x_node
      result = (c @ b).tensor
      return result + bias_var
                       
      result = tf.vectorized_map(
        lambda vec: f(vec, self.a_var, self.b_var, self.bias), inputs)
   return tf.nn.swish(tf.reshape(result, (-1, 64)))


Dense = tf.keras.layers.Dense
fc_model = tf.keras.Sequential(
 [
     tf.keras.Input(shape=(2,)),
     Dense(64, activation=tf.nn.swish),
     Dense(64, activation=tf.nn.swish),
     Dense(1, activation=None)])

fc_model.summary()


tn_model = tf.keras.Sequential(
    [
     tf.keras.Input(shape=(2,)),
     Dense(64, activation=tf.nn.swish),
      TNLayer(),
     Dense(1, activation=None)])

tn_model.summary()



X = np.concatenate([np.random.randn(20, 2) + np.array([3, 3]), 
             np.random.randn(20, 2) + np.array([-3, -3]), 
             np.random.randn(20, 2) + np.array([-3, 3]), 
             np.random.randn(20, 2) + np.array([3, -3]),])

#Y = np.concatenate([np.ones((40)), -np.ones((40))])

tn_model.compile(optimizer="adam", loss="mean_squared_error")
tn_model.fit(X, Y, epochs=300, verbose=1)

h = 1.0
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
