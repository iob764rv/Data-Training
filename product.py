import pytest
import numpy as np
import tensorflow as tf
import tensornetwork as tn
from examples.wavefunctions import wavefunctions


@pytest.mark.parametrize("num_sites", [2, 3, 4])
#def test_expval(num_sites):
 # op = np.kron(np.array([[1.0, 0.0], [0.0, -1.0]]), np.eye(2)).reshape([2] * 4)
  #op = tf.convert_to_tensor(op)
#  for j in range(num_sites):
 #   psi = np.zeros([2] * num_sites)
  #  psi_vec = psi.reshape((2**num_sites,))
   # psi_vec[2**j] = 1.0
 #   #psi = tf.convert_to_tensor(psi)
  #  for i in range(num_sites):
   #   res = wavefunctions.expval(psi, op, i, pbc=True)
    #  if i == num_sites - 1 - j:
      #  np.testing.assert_allclose(res, -1.0)
     # else:
        np.testing.assert_allclose(res, 1.0)

#tensor
from functools import reduce

def inner(psi1, psi2):
  return tf.reduce_sum(tf.math.conj(psi1) * psi2)

  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)

  n_res = tensornetwork.contract_between(
      n_op, n_psi, output_edge_order=site_edges)

  return n_res.tensor


def _apply_op_network(site_edges, op, n1, pbc=False):
  
  N = len(site_edges)
  op_sites = len(op.shape) // 2
  n_op = tensornetwork.Node(op, backend="tensorflow")
  
  for m in range(op_sites):
    target_site = (n1 + m) % N if pbc else n1 + m
    tensornetwork.connect(n_op[op_sites + m], site_edges[target_site])
    site_edges[target_site] = n_op[m]
  return site_edges, n_op


def expval(psi, op, n1, pbc=False):

  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)
  n_op_psi = n_op @ n_psi
  n_psi_conj = tensornetwork.Node(tf.math.conj(psi), backend="tensorflow")
  
  for i in range(len(site_edges)):
    tensornetwork.connect(site_edges[i], n_psi_conj[i])
  res = n_psi_conj @ n_op_psi
  
  return res.tensor

def evolve_trotter(psi,
                   H,
                   step_size,
                   num_steps,
                   euclidean=False,
                   callback=None):

  num_sites = len(psi.shape)
  layers = trotter_prepare_gates(H, step_size, num_sites, euclidean)
  return _evolve_trotter_gates(
      psi, layers, step_size, num_steps, euclidean=euclidean, callback=callback)


def apply_circuit(psi, layers):
  num_sites = len(psi.shape)

  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
  nodes = [n_psi]

  for gates in layers:
    skip = 0
    for n in range(num_sites):
      if n < len(gates):
        gate = gates[n]
      else:
        gate = None

      if skip > 0:
        if gate is not None:
          raise ValueError(
              
        skip -= 1
      elif gate is not None:
        site_edges, n_gate = _apply_op_network(site_edges, gate, n)
        nodes.append(n_gate)

       
        op_sites = len(gate.shape) // 2
        skip = op_sites - 1

 
  n_psi = reduce(tensornetwork.contract_between, nodes)
  n_psi.reorder_edges(site_edges)

  return n_psi.tensor
