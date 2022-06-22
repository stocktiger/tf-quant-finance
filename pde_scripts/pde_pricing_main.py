import tensorflow as tf
import tf_quant_finance as tff 
import datetime
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from local_vol import get_tf_volsurface_from_bastion, get_vol_surface_bastion
from options_pde import american_option, my_barrier_option
from utils import query_spot_price

def option_param(number_of_options, dtype, seed=42):
  """ Function to generate volatilities, rates, strikes """
  np.random.seed(seed)
  if number_of_options > 1:
    volatility = tf.random.uniform(shape=(number_of_options, 1),
                                   dtype=dtype) * 0.1 + 0.3
    # Random risk free rate between 0 and 0.2.
    risk_free_rate = tf.constant(
      np.random.rand(number_of_options, 1) * 0.05, dtype)
    # Random strike between 20 and 120.
    strike = tf.constant(
      np.random.rand(number_of_options, 1) * 100 + 50, dtype)
  else:
    volatility = tf.constant([0.8], dtype)
    risk_free_rate = tf.constant([0.0], dtype)
    strike = tf.constant([30000], dtype)
  return volatility, risk_free_rate, strike

def check_numerical_stability(s_min, s_max, number_grid_points, risk_free_rate, volatility, time_delta):
  s = np.linspace(s_min, s_max, num=number_grid_points)
  delta_s = (s_max-s_min)/(number_grid_points-1)
  eta = 2* risk_free_rate * s / (2*delta_s)
  con_1 = tf.less(delta_s**2 / s**2, volatility**2 /eta)
  con_2 = tf.less(1/(0.5*time_delta), volatility**2 * (s**2)/(delta_s**2) + risk_free_rate)
  stability_con = con_1 & con_2
  pd.DataFrame((stability_con).numpy().astype(int)).plot(title="stability plot")
  plt.show()
  return stability_con

def plot_results(path, dfs, names):
    for file in os.listdir(path):
      names.append("_".join(file.split()[5:]))
      dfs.append(pd.read_csv(os.path.join(path, file), index_col=0).set_index("spot"))
    z = pd.concat(dfs, axis=1).set_axis(names, axis=1)
    z.to_csv("summary.csv")
    z[[col for col in z.columns if "contin" in col]].plot()
    plt.show()
    z[[col for col in z.columns if "bermu" in col]].plot()
    plt.show()

if __name__ == "__main__":
    
    ######## for plottings comparison ########
    path = r"_resultssss\barrier30000strike15000"
    dfs = []
    names = []
    # plot_results(path, dfs, names)
    ######## for plottings comparison ########

    
    
    #@title Price multiple American Call options at a time

    number_of_options = 1 #@param

    time_delta = 0.1/365 #0.0027397260273972603

    dtype = tf.float32

    DAY_REF = datetime.date(2022, 6, 8)
    EXPIRE_DATE = datetime.date(2022, 12, 9)
    KO_DATES = list(reversed([
      datetime.date(2022, 7, 11),
      datetime.date(2022, 8, 9),
      datetime.date(2022, 9, 9),
      datetime.date(2022, 10, 10),
      datetime.date(2022, 11, 9)
    ]))
    start_date = tff.datetime.dates_from_datetimes(DAY_REF)
    expiry_date = tff.datetime.dates_from_datetimes(EXPIRE_DATE)
    ko_dates = tff.datetime.dates_from_datetimes(KO_DATES)
    expiry = float(tff.datetime.daycount_actual_365_fixed(start_date=start_date, end_date=expiry_date, dtype=dtype).numpy())
    price_range = 1, 65000
    risk_free_rate = tf.constant([0.0], dtype)
    strike = tf.constant([14896], dtype)
    barrier = tf.constant([29792], dtype)
    
    # ko_check_time = tf.range(1.5/12, 6.5/12, 1/12, dtype=dtype)             ### Bermuda Check
    # ko_check_time = tf.reshape(tf.convert_to_tensor((), dtype=dtype), (0,)) ### CONTINUOUS KO CHECK
    ko_check_time = tff.datetime.daycount_actual_365_fixed(start_date=ko_dates, end_date=expiry_date, dtype=dtype)  ### FCNs
    
    vanilla_type = "put"
    ko_type = "up-and-out"
    number_grid_points = 5*int(5 * ((expiry/time_delta)**0.5)) #### M = 5 * sqrt(N) so the grid is large enough to avoid feeling the boundaries
    initial_spot = tf.constant([[query_spot_price("BTC")]], dtype=dtype)
    

    ######################### Volatility setup #########################
    ### Use our fitted vol surface
    volsurface = get_tf_volsurface_from_bastion(dtype=dtype)
    @tf.function
    def get_vol(*arg, **kwarg):
      return volsurface.volatility(*arg, **kwarg)
    volatility = get_vol
    
    ### Constant vol
    # volatility = tf.constant([0.65], dtype)

    ### Constant vol surface across time
    # s = np.linspace(price_range[0], price_range[1], num=number_grid_points)
    # volatility = tf.convert_to_tensor(get_vol_surface_bastion(s), dtype=dtype)
    ######################### Volatility setup #########################


    # check_numerical_stability(price_range[0], price_range[1], number_grid_points, risk_free_rate, volatility, time_delta)

    # # Build a graph to compute prices of the American Options.
    # estimate, grid_locations = american_option(
    #     time_delta=time_delta,
    #     expiry=expiry,
    #     number_grid_points=number_grid_points,
    #     volatility=volatility,
    #     risk_free_rate=risk_free_rate,
    #     strike=strike,
    #     final_payoff_fn=put_final_payoff_fn,
    #     dtype=dtype)

    with tf.device('/GPU:0'):
      estimate, grid_locations = my_barrier_option(
          number_grid_points=number_grid_points,
          price_range=price_range,
          time_delta=time_delta,
          strike=strike,
          barrier=barrier,
          vanilla_type=vanilla_type,
          ko_type=ko_type,
          ko_check_time=ko_check_time, ### tensor array of KO check time in years
          volatility=volatility,
          risk_free_rate=risk_free_rate,
          expiry=expiry,
          initial_spot=initial_spot,
          volsurface=volsurface,
          dtype=dtype)

    # Convert to numpy for plotting
    estimate = estimate.numpy()
    grid_locations = grid_locations.numpy()
    # Output the pricing
    save_path = "_results/tmp"
    os.makedirs(save_path, exist_ok=True)
    pd.DataFrame({"spot": grid_locations, "value": estimate}).to_csv(f"{pd.Timestamp.now().date()} American {ko_type} {vanilla_type} Options.csv")

    
    # Prepare data for plotting 
    options = [x + 1 for x in range(number_of_options) for _ in range(number_grid_points)]
    plot_data = pd.DataFrame({
        'Spot': list(np.ndarray.flatten(grid_locations)) * number_of_options, 
        'Price': estimate.flatten(),
        'Option': options})


    # Plot
    plt.figure(figsize=(10, 8))
    sns.set(style="darkgrid")
    sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
    plot = sns.lineplot(x="Spot", y="Price", hue="Option",
                        data=plot_data,
                        palette=sns.color_palette()[:number_of_options],
                        legend=False)
    plot.axes.set_title(f"Price/Spot for {number_of_options} American {ko_type} {vanilla_type} Options",
                        fontsize=25)
    xlabel = plot.axes.get_xlabel()
    ylabel = plot.axes.get_ylabel()
    plot.axes.set_xlabel(xlabel, fontsize=20)
    plot.axes.set_ylabel(ylabel, fontsize=20)
    plt.show()