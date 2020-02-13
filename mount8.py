import numpy as np
import jax
import tensornetwork as tn

a = tn.Node(np.ones(10))
# Either tensorflow tensors or numpy arrays are fine.
b = tn.Node(np.ones(10))
