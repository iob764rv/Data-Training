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
#  op_sites = len(op.shape) // 2
 # n_op = tensornetwork.Node(op, backend="tensorflow")
  #for m in range(op_sites):
#    target_site = (n1 + m) % N if pbc else n1 + m
 #   tensornetwork.connect(n_op[op_sites + m], site_edges[target_site])
  #  site_edges[target_site] = n_op[m]
  return site_edges, n_op
