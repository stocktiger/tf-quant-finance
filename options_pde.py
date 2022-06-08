from pickle import FALSE
import tensorflow as tf
# Import for Tensorflow Quant Finance

import tf_quant_finance as tff 
import numpy as np

# dtype = np.float64
# discount_rates = np.array([0.0, .08])
# dividend_rates = np.array([0.0, .04])
# spots = np.array([100., 100.])
# strikes = np.array([50., 90.])
# barriers = np.array([60., 95.])
# rebates = np.array([0., 3.])
# volatilities = np.array([.3, .25])
# expiries = np.array([6.5/12, .5])
# is_barrier_down = np.array([False, False])
# is_knock_out = np.array([True, False])
# is_call_option = np.array([False, True])

# price = tff.black_scholes.barrier_price(
#   discount_rates,
#   dividend_rates,
#   spots, 
#   strikes,
#   barriers, 
#   rebates, 
#   volatilities,
#   expiries, 
#   is_barrier_down, 
#   is_knock_out, 
#   is_call_option)
# tf.print(price)

# Shortcut alias
pde = tff.math.pde
# option_price = tff.black_scholes.option_price
# implied_vol = tff.black_scholes.implied_vol

@tf.function
def isclose(a, b, rel_tol=tf.constant(0.0, dtype=tf.float64), abs_tol=tf.constant(0.0, dtype=tf.float64)):
    return tf.math.less_equal(tf.math.abs(a-b), tf.math.maximum(rel_tol * tf.math.maximum(tf.math.abs(a), tf.math.abs(b)), abs_tol))

# tf.function decorator makes the function faster in graph mode.
@tf.function
def my_barrier(number_grid_points, # python int
                    price_range, # python tuple of float
                    time_delta, # python float
                    strike,
                    barrier,
                    vanilla_type,
                    ko_type, #### python str: either up/down-and-in/out should be used e.g. "up-and-out"
                    ko_check_time, ### tensor array of KO check time in years
                    volatility,
                    risk_free_rate,
                    expiry,
                    dtype=tf.float64):
  
  if ko_type in ["up-and-in", "down-and-in"]:
    print(f"{ko_type} is not supported right now.\n Please evaluate vanilla - equivalent knock-out instead.")
    return
  
  # Define the coordinate grid
  s_min, s_max = price_range
  # s_min = min(0.01, s_min)
  # s_max = max(300, s_max)
  grid = pde.grids.uniform_grid(minimums=[s_min],
                                maximums=[s_max],
                                sizes=[number_grid_points],
                                dtype=dtype)
  s_delta = (s_max-s_min)/(number_grid_points-1)

  # Define the values grid for the final condition
  s = grid[0]  ## list of stock prices at each time point
  # final_values_grid = final_payoff_fn(s, strike, volatility)
  if vanilla_type=="put":
    final_values_grid = tf.nn.relu(strike - s)
  else:
    final_values_grid = tf.nn.relu(s - strike)
  if ko_type=="up-and-out":
    final_values_grid = tf.where(tf.greater_equal(s, barrier), tf.constant(0.0, dtype=dtype), final_values_grid)
  elif ko_type=="down-and-out":
    final_values_grid = tf.where(tf.less_equal(s, barrier), tf.constant(0.0, dtype=dtype), final_values_grid)


  # Define the PDE coefficient functions
  def second_order_coeff_fn(t, grid):
    del t
    #### TODO local vol replace volatility at each timepoint
    #### TODO volatility = volatility(t, S)
    s = grid[0]

    return [[volatility ** 2 * s ** 2 / 2]]

  def first_order_coeff_fn(t, grid):
    del t
    s = grid[0]
    return [risk_free_rate * s]

  def zeroth_order_coeff_fn(t, grid):
    del t, grid
    return -risk_free_rate

  # Define the boundary conditions
  @pde.boundary_conditions.dirichlet
  def vanilla_lower_boundary_fn(t, grid):
    del t, grid
    return tf.constant(0.0, dtype=dtype)
s
  @pde.boundary_conditions.dirichlet
  def call_upper_boundary_fn(t, grid):
    del grid
    return tf.squeeze(s_max - strike * tf.exp(-risk_free_rate * (expiry - t)))

  @pde.boundary_conditions.dirichlet
  def put_upper_boundary_fn(t, grid):
    del grid
    return tf.squeeze(strike * tf.exp(-risk_free_rate * (expiry - t)) - 0.0)

  if vanilla_type=="put":
    lower_boundary_fn = put_upper_boundary_fn
    upper_boundary_fn = vanilla_lower_boundary_fn
  else:
    upper_boundary_fn = call_upper_boundary_fn
    lower_boundary_fn = vanilla_lower_boundary_fn

  # In order to price American option one needs to set option values to 
  # V(x) := max(V(x), max(x - strike, 0)) after each iteration
  def values_transform_fn(t, grid, values):
    # return grid, values

    # tf.print(tf.math.reduce_any(isclose(ko_check_time, t)), t)
    # check t and discretely maximize the binary?
    if tf.size(ko_check_time)==0 or \
     tf.math.reduce_any(isclose(ko_check_time, t, tf.constant(0.0, dtype=dtype), tf.constant(time_delta/2, dtype=dtype))):
      tf.print(tf.strings.format("KO check @ {} year", (tf.round(t*1000)/1000)))
      s = grid[0]
      if ko_type=="up-and-out":
        mod_values = tf.where(tf.greater_equal(s, barrier), tf.constant(0.0, dtype=dtype), values)
      # elif ko_type=="up-and-in":
      #   mod_values = tf.where(tf.greater_equal(s, barrier), tf.constant(0.0, dtype=dtype), values)
      elif ko_type=="down-and-out":
        mod_values = tf.where(tf.less_equal(s, barrier), tf.constant(0.0, dtype=dtype), values)
      # elif ko_type=="down-and-in":
      #   mod_values = tf.where(tf.greater_equal(s, barrier), tf.constant(0.0, dtype=dtype), values)
      return grid, mod_values
    else:
      return grid, values

  # Solve
  estimate_values, estimate_grid, _, _ = \
    pde.fd_solvers.solve_backward(
      start_time=expiry,
      end_time=0,
      values_transform_fn=values_transform_fn,
      coord_grid=grid,
      values_grid=final_values_grid,
      time_step=time_delta,
      boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
      second_order_coeff_fn=second_order_coeff_fn,
      first_order_coeff_fn=first_order_coeff_fn,
      zeroth_order_coeff_fn=zeroth_order_coeff_fn,
      dtype=dtype
    )
  return estimate_values, estimate_grid[0]

# tf.function decorator makes the function faster in graph mode.
@tf.function
def american_option(number_grid_points,
                    time_delta,
                    strike,
                    volatility,
                    risk_free_rate,
                    expiry,
                    final_payoff_fn,
                    dtype=tf.float64):
  """ Computes American Call options prices.

  Args:
    number_grid_points: A Python int. Number of grid points for the finite
      difference scheme.
    time_delta: A Python float. Grid time discretization parameter.
    strike: A real `Tensor` of shape `(number_of_options, 1)`.
      Represents the strikes of the underlying American options. 
    volatility: A real `Tensor` of shape `(number_of_options, 1)`.
      Represents the volatilities of the underlying American options. 
    risk_free_rate: A real `Tensor` of shape `(number_of_options, 1)`.
      Represents the risk-free interest rates associated with the underlying
      American options.
    expiry: A Python float. Expiry date of the options. If the options
      have different expiries, volatility term has to adjusted to
      make expiries the same.
    dtype: Optional `tf.dtype` used to assert dtype of the input `Tensor`s.

  Returns:
    A tuple of the estimated option prices of shape
    `(number_of_options, number_grid_points)` and the corresponding `Tensor` 
    of grid locations of shape `(number_grid_points,)`.
  """
  # Define the coordinate grid
  s_min = 0.01
  s_max = 999999.
  grid = pde.grids.uniform_grid(minimums=[s_min],
                                maximums=[s_max],
                                sizes=[number_grid_points],
                                dtype=dtype)

  # Define the values grid for the final condition
  s = grid[0] ## list
  final_values_grid = final_payoff_fn(s, strike, volatility)

  # Define the PDE coefficient functions
  def second_order_coeff_fn(t, grid):
    s = grid[0]
    return [[volatility ** 2 * s ** 2 / 2]]

  def first_order_coeff_fn(t, grid):
    del t
    s = grid[0]
    return [risk_free_rate * s]

  def zeroth_order_coeff_fn(t, grid):
    del t, grid
    return -risk_free_rate

  # Define the boundary conditions
  @pde.boundary_conditions.dirichlet
  def lower_boundary_fn(t, grid):
    del t, grid
    return tf.constant(0.0, dtype=dtype)

  @pde.boundary_conditions.dirichlet
  def upper_boundary_fn(t, grid):
    del grid
    return tf.squeeze(s_max - strike * tf.exp(-risk_free_rate * (expiry - t)))

  # In order to price American option one needs to set option values to 
  # V(x) := max(V(x), max(x - strike, 0)) after each iteration
  def values_transform_fn(t, grid, values):

    del t
    s = grid[0]
    values_floor = tf.nn.relu(s - strike)
    return grid, tf.maximum(values, values_floor)

  # Solve
  estimate_values, estimate_grid, _, _ = \
    pde.fd_solvers.solve_backward(
      start_time=expiry,
      end_time=0,
      values_transform_fn=values_transform_fn,
      coord_grid=grid,
      values_grid=final_values_grid,
      time_step=time_delta,
      boundary_conditions=[(lower_boundary_fn, upper_boundary_fn)],
      second_order_coeff_fn=second_order_coeff_fn,
      first_order_coeff_fn=first_order_coeff_fn,
      zeroth_order_coeff_fn=zeroth_order_coeff_fn,
      dtype=dtype
    )
  return estimate_values, estimate_grid[0]
