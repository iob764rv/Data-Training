import tensorflow as tf


#def trotter_prepare_gates(H, step_size, num_sites, euclidean):
 # if not len(H) == num_sites - 1:
  #  raise ValueError("Number of H terms must match number of sites - 1.")
  #  step_size = tf.cast(step_size, tf.float64)  # must be real
  #step_size = tf.cast(step_size, H[0].dtype)

 #   if euclidean:
  #  step_size = -1.0 * step_size
  #else:
   # step_size = 1.j * step_size
  #  eH = []
  #for h in H:
   # if len(h.shape) != 4:
    #  raise ValueError("H must be nearest-neighbor.")
