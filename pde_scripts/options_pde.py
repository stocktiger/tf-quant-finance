import tensorflow as tf
import numpy as np
# Import for Tensorflow Quant Finance

import tf_quant_finance as tff 
from local_vol import dupire_local_volatility

# Shortcut alias
pde = tff.math.pde
# option_price = tff.black_scholes.option_price
# implied_vol = tff.black_scholes.implied_vol

@tf.function
def isclose(a, b, rel_tol=tf.constant(0.0, dtype=tf.float64), abs_tol=tf.constant(0.0, dtype=tf.float64)):
    return tf.math.less_equal(tf.math.abs(a-b), tf.math.maximum(rel_tol * tf.math.maximum(tf.math.abs(a), tf.math.abs(b)), abs_tol))

# tf.function decorator makes the function faster in graph mode.
@tf.function()
def my_barrier_option(number_grid_points, # python int
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
                    initial_spot,
                    volsurface,
                    annual_coupon_rate = 0.19,
                    dividend_yield = 0.,
                    dtype=tf.float64):
  
  if ko_type in ["up-and-in", "down-and-in"]:
    print(f"{ko_type} is not supported right now.\n Please evaluate vanilla minuss equivalent knock-out instead.")
    return
  
  # Prepare the volsurface for the local vol model
  discount_factor_fn = lambda t: tf.math.exp(-risk_free_rate * t)
  dividend = tf.convert_to_tensor([dividend_yield], dtype=dtype) #### HARDCODE

  # Define the coordinate grid
  s_min, s_max = price_range
  grid = pde.grids.uniform_grid(minimums=[s_min],
                                maximums=[s_max],
                                sizes=[number_grid_points],
                                dtype=dtype)
  s_delta = (s_max-s_min)/(number_grid_points-1)

  # Define the values grid for the final condition
  s = grid[0]  ## list of stock prices at each time point
  s_ = tf.reshape(s, [-1, 1])

  if vanilla_type=="put":
    final_values_grid = 2*tf.nn.relu(strike - s) - tf.squeeze(initial_spot*annual_coupon_rate*6/12)
  else:
    final_values_grid = 2*tf.nn.relu(s - strike) - tf.squeeze(initial_spot*annual_coupon_rate*6/12)
  if ko_type=="up-and-out":
    final_values_grid = tf.where(tf.greater_equal(s, barrier), tf.constant(0.0, dtype=dtype), final_values_grid)
  elif ko_type=="down-and-out":
    final_values_grid = tf.where(tf.less_equal(s, barrier), tf.constant(0.0, dtype=dtype), final_values_grid)

  # Define the PDE coefficient functions
  def second_order_coeff_fn(t, grid):
    s = grid[0]
    
    ### Constant vol
    # return [[volatility ** 2 * s ** 2 / 2]]

    ### Constant vol surface across time
    # vol_from_ini_volsurface = volatility(strike=tf.reshape(s, (1,-1)), 
    #                                                 expiry_times=tf.reshape(tf.expand_dims(expiry - t, 0),(1,-1)))
    # vol_from_ini_volsurface = tf.squeeze(vol_from_ini_volsurface)
    # return [[vol_from_ini_volsurface ** 2 * s ** 2 / 2]]

    ### Local Vol
    t_ = tf.expand_dims(expiry - t, 0)
    # t_ = tf.expand_dims(t, 0)

    local_vol = dupire_local_volatility(t_, 
                                s_, 
                                initial_spot, 
                                volsurface.volatility, 
                                discount_factor_fn, 
                                dividend)
    local_vol = tf.squeeze(local_vol)
    return [[local_vol ** 2 * s ** 2 / 2]]

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

  # KO checkings
  def values_transform_fn(t, grid, values):
    # tf.print(tf.math.reduce_any(isclose(ko_check_time, t)), t)
    if tf.size(ko_check_time)==0 or \
     tf.math.reduce_any(isclose(ko_check_time, t, tf.constant(0.0, dtype=dtype), tf.constant(time_delta/2, dtype=dtype))):
      s = grid[0]
      num_period = tf.squeeze(tf.where(isclose(ko_check_time, t, tf.constant(0.0, dtype=dtype), tf.constant(time_delta/2, dtype=dtype))))
      KO_value = tf.fill(values.shape, tf.squeeze(-1*initial_spot*annual_coupon_rate*((tf.cast(num_period, dtype=dtype)+1.0)/12.0)))
      tf.print(tf.strings.format("Month {} KO check @ {} year, pay {}", (num_period, tf.round(t*1000)/1000, KO_value)))
      # KO_value = tf.constant(0.0, dtype=dtype)
      mod_values = None
      if ko_type=="up-and-out":
        mod_values = tf.where(tf.greater_equal(s, barrier), KO_value, values)
      elif ko_type=="down-and-out":
        mod_values = tf.where(tf.less_equal(s, barrier), KO_value, values)
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
