import numpy as np
import jax
import tensornetwork as tn

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
