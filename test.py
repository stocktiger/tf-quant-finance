import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from options_pde import american_option, my_barrier

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

def check_stability(s_min, s_max, number_grid_points, risk_free_rate, volatility, time_delta):
  s = np.linspace(s_min, s_max, num=number_grid_points)
  delta_s = (s_max-s_min)/(number_grid_points-1)
  eta = 2* risk_free_rate * s / (2*delta_s)
  con_1 = tf.less(delta_s**2 / s**2, volatility**2 /eta)
  con_2 = tf.less(1/(0.5*time_delta), volatility**2 * (s**2)/(delta_s**2) + risk_free_rate)
  stability_con = con_1 & con_2
  pd.DataFrame((stability_con).numpy().astype(int)).plot(title="stability plot")
  plt.show()
  return stability_con

if __name__ == "__main__":
    #@title Price multiple American Call options at a time

    number_of_options = 1 #@param

    time_delta = 0.1/365 #0.0027397260273972603

    # expiry = 1


    dtype = tf.float64

    # spot = 110  + tf.random.uniform(shape=[number_of_options, 1], dtype=dtype)

    # Generate volatilities, rates, strikes
    volatility, risk_free_rate, strike = option_param(number_of_options, dtype)

    expiry = 6.5/12
    price_range = 0.001, 400
    volatility = tf.constant([0.3], dtype)
    risk_free_rate = tf.constant([0.0], dtype)
    strike = tf.constant([100], dtype)
    barrier = tf.constant([120], dtype)
    ko_check_time = tf.range(1.5/12, 6.5/12, 1/12, dtype=dtype)
    # ko_check_time = tf.reshape(tf.convert_to_tensor((), dtype=dtype), (0,)) ### ALL
    vanilla_type = "put"
    ko_type = "up-and-out"
    number_grid_points = max(102400, int(5 * ((expiry/time_delta)**0.5))) #### M = 5 * sqrt(N) so the grid is large enough to avoid feeling the boundaries

    # check_stability(price_range[0], price_range[1], number_grid_points, risk_free_rate, volatility, time_delta)

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

    estimate, grid_locations = my_barrier(
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
        dtype=dtype)

    # Convert to numpy for plotting
    estimate = estimate.numpy()
    grid_locations = grid_locations.numpy()
    import pandas as pd
    pd.DataFrame({"spot": grid_locations, "value": estimate}).to_csv(f"{pd.Timestamp.now().date()} American {ko_type} {vanilla_type} Options.csv")

    from matplotlib import pyplot as plt
    import pandas as pd
    import seaborn as sns

    
    #@title Price/spot plot for the American Call options
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