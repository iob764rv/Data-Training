#tensor
  n_psi = tensornetwork.Node(psi, backend="tensorflow")
  site_edges = n_psi.get_all_edges()
  site_edges, n_op = _apply_op_network(site_edges, op, n1, pbc)

  n_res = tensornetwork.contract_between(
      n_op, n_psi, output_edge_order=site_edges)

  return n_res.tensor
