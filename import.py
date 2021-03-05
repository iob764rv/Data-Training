import tensorflow as tf
from functools import reduce
import sys
import tensornetwork
from examples.wavefunctions.trotter import trotter_prepare_gates


def trotter_prepare_gates(H, step_size, num_sites, euclidean):
  if not len(H) == num_sites - 1:
    raise ValueError("Number of H terms must match number of sites - 1.")
    step_size = tf.cast(step_size, tf.float64)  # must be real
  step_size = tf.cast(step_size, H[0].dtype)

    if euclidean:
    step_size = -1.0 * step_size
  else:
    step_size = 1.j * step_size
    eH = []
 for h in H:
    if len(h.shape) != 4:
      raise ValueError("H must be nearest-neighbor.")
        h_shp = tf.shape(h)
    h_r = tf.reshape(h, (h_shp[0] * h_shp[1], h_shp[2] * h_shp[3]))
    eh_r = tf.linalg.expm(step_size * h_r)
    eH.append(tf.reshape(eh_r, h_shp))
    
  eh_even = [None] * num_sites
  eh_odd = [None] * num_sites
  for (n, eh) in enumerate(eH):
  
  if n % 2 == 0:
      eh_even[n] = eh
    else:
      eh_odd[n] = eh

  return [eh_even, eh_odd]


def inner(psi1, psi2):
  """Computes the inner product <psi1|psi2>.
    inner_product: The vector inner product.
  """
  return tf.reduce_sum(tf.math.conj(psi1) * psi2)

def apply_op(psi, op, n1, pbc=False):
    n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
#
  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)

  n_res = tensornetwork.contract_between(
      n_op, n_psi, output_edge_order=site_edges)

  return n_res.tensor

def _evolve_trotter_gates_defun(psi,
                                layers,
                                step_size,
                                num_steps,
                                euclidean=False,
                                callback=None):
  return _evolve_trotter_gates(
  #    psi, layers, step_size, num_steps, euclidean=euclidean, callback=callback)
    
#    def _evolve_trotter_gates(psi,
 #                         layers,
  #                        step_size,
   #                       num_steps,
    #                      euclidean=False,
   #                       callback=None):
  """Evolve an initial wavefunction psi via gates specified in `layers`.
  If the evolution is euclidean, the wavefunction will be normalized
  after each step.
  """
  #t = 0.0
  #for i in range(num_steps):
   # psi = apply_circuit(psi, layers)
   # if euclidean:
    #  psi = tf.divide(psi, tf.norm(psi))
   # t += step_size
  #  if callback is not None:
 #     callback(psi, t, i)

#  return psi, t


