import numpy as np
import jax
import tensornetwork as tn

def one_edge_at_a_time(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  edge1 = node1[0] ^ node2[0]
  edge2 = node1[1] ^ node2[1]
  tn.contract(edge1)
  result = tn.contract(edge2)
  return result.tensor

def use_contract_between(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  node1[0] ^ node2[0]
  node1[1] ^ node2[1]
 # This is the same as 
 # tn.contract_between(node1, node2)
  result = node1 @ node2
  
 def use_contract_parallel(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  edge = node1[0] ^ node2[0]
  node1[1] ^ node2[1]
  result = tn.contract_parallel(edge)
   # is fully contracted.
  return result.tensor

def use_contract_parallel(a, b):
  node1 = tn.Node(a)
  node2 = tn.Node(b)
  edge = node1[0] ^ node2[0]
  node1[1] ^ node2[1]
  result = tn.contract_parallel(edge)
  return result.tensor
#fully contracted

def calculate_abc_trace(a, b, c):
  an = tn.Node(a)
  bn = tn.Node(b)
  cn = tn.Node(c)
  
  an[1] ^ bn[0]
  bn[1] ^ cn[0]
  cn[1] ^ an[0]
  return (an @ bn @ cn).tensor
  #####

a = tn.Node(np.ones(10))
# Either tensorflow tensors or numpy arrays are fine.
b = tn.Node(np.ones(10))
edge = a[0] ^ b[0]
c = tn.contract(edge)
print(c.tensor)
a = tn.Node(np.eye(2))


c = tn.contract_between(a, b)
c = a @ b
c = tn.contract_parallel(edge)

u_s, vh_s, trun_error = tn.split_node(node, left_edges, right_edges)
u, s, vh, trun_error = tn.split_node_full_svd(node, left_edges, right_edges)
node = tn.Node(np.eye(2), name="Identity Matrix")

trace_edge = a[0] ^ a[1]
print("Is a[0] dangling?:", a[0].is_dangling())

a = tn.Node(np.eye(2), axis_names=['alpha', 'beta'])
edge = a['alpha'] ^ a['beta']
result = tn.contract(edge)
print(result.tensor)

#result is 2.0
a = np.ones((1000, 1000))
b = np.ones((1000, 1000))

%timeit one_edge_at_a_time(a, b)
#print("Running use_cotract_between")
%timeit use_contract_between(a, b)
##runs one edge at  a time

a = tn.Node(np.ones((2, 2, 2)))
b = tn.Node(np.ones((2, 2, 2)))
c = tn.Node(np.ones((2, 2, 2)))
d = tn.Node(np.ones((2, 2, 2)))
# Make the network fully connected.

a[0] ^ b[0]
a[1] ^ c[1]
a[2] ^ d[2]
b[1] ^ d[1]
b[2] ^ c[2]
c[0] ^ d[0]

#"greedy" contraction algorithm
nodes = tn.reachable(a)
result = tn.contractors.greedy(nodes)
#print(result.tensor)
#result is 64.0
ones = np.ones((2, 2, 2))

tn.ncon([ones, ones, ones, ones], 
        [[1, 2, 4], 
         [1, 3, 5], 
         [2, 3, 6],
         [4, 5, 6]])

ones = np.ones((2, 2))
tn.ncon([ones, ones], [[-1, 1], [1, -2]])

diagonal_array =np.array([[2.0, 0.0, 0.0],
                           [0.0, 2.5, 0.0],
                           [0.0, 0.0, 1.5]]) 
a = tn.Node(diagonal_array)
print("Contraction of U and V*:")
print((u @ vh).tensor)
u, vh, truncation_error = tn.split_node(
    a, left_edges=[a[0]], right_edges=[a[1]], max_singular_values=2)
#a = tn.Node(diagonal_array)


print(truncation_error)
a = np.ones((4096, 4096))
b = np.ones((4096, 4096))
c = np.ones((4096, 4096))

tn.set_default_backend("numpy")
%timeit np.array(calculate_abc_trace(a, b, c))
tn.set_default_backend("jax")
########
tf.enable_v2_behavior()
tn.set_default_backend("tensorflow")


class TNLayer(tf.keras.layers.Layer):
 #  super(TNLayer, self).__init__()
    self.a_var = tf.Variable(tf.random.normal(
            shape=(8, 8, 2), stddev=1.0/16.0),
             name="a", trainable=True)
    self.b_var = tf.Variable(tf.random.normal(shape=(8, 8, 2), stddev=1.0/16.0),
                             name="b", trainable=True)
    self.bias = tf.Variable(tf.zeros(shape=(8, 8)), name="bias", trainable=True)
    
  def call(self, inputs):
    def f(input_vec, a_var, b_var, bias_var):
      input_vec = tf.reshape(input_vec, (8,8))
