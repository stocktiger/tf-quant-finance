'''
    Contains the MC simulation implementation using Bastion Vol
'''
import datetime
import time
import os

# Import all the libraries as required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_quant_finance as tff
from tf_quant_finance.black_scholes import option_price
from tf_quant_finance.experimental.pricing_platform.framework.market_data.volatility_surface import \
    VolatilitySurface

from support.LocalVolModel import (LocalVolatilityModel, _dupire_local_volatility_iv)
from support.RestClient import RestClient

os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=C:/Users/michael/Anaconda3/envs/googletf/Library/bin/"

CLIENT = RestClient("https://defi-bot.bastioncb.com:1443/")

def generate_input_data(today_date, reference_date, dtype=tf.float32):
    ''' puts everything into tensors where necessary'''
    # Specify date
    valuation_date = tff.datetime.dates_from_datetimes([today_date])
    # Query approximately 1 year of volsurface data on a daily interval
    dates = [reference_date +
                     datetime.timedelta(days=i) for i in range(1, 30*12)]
    # SET expiries as a (1 x Maturities) tensor
    expiries = tff.datetime.dates_from_datetimes(dates).reshape((1, -1))
    expiry_times = tff.datetime.daycount_actual_365_fixed(
            start_date=valuation_date, end_date=expiries, dtype=dtype)

    # Construct the strike space to query
    dk = 250
    strikes = [10000 + dk*i for i in range(60000//dk)]
    return valuation_date, dates, expiries, expiry_times, strikes

def get_bastion_data(dates, strikes):
    ''' Get the data from Bastion's volsurface service '''
    date_strings = ",".join([date.strftime('%d%b%y').upper() for date in dates])
    strikes_strings = ",".join([str(strike) for strike in strikes])
    params = {"symbols": "BTC", "maturities": date_strings,
                        "strikes": strikes_strings}

    # Obtain IVs, Forwards, Spot from volsurface:
    # dict: { 'iv': {maturity: {price: iv, ... }, ... },
    #                 'forward_price': {maturity: price, ...},
    #                    'spot': price }
    response = CLIENT.get_endpoint('volsurface/forward_price', params)
    response_ivs, response_forwards, SPOT_REF = response[
            "iv"], response["forward_price"], response["spot"]

    print("SPOT REFERENCE", SPOT_REF)
    return response_ivs, response_forwards, SPOT_REF

def construct_volsurface(valuation_date, expiries, expiry_times, strikes, bastion_vols, dtype=tf.float32):
    ''' construct volsurface from input data and return VolatilitySurface object'''

    # SET strikes as (1 x Maturities x Strikes) tensor
    strikes = [expiries.shape[1]*[strikes]]
    strikes = tf.constant(strikes, dtype=dtype)

    input_vols = pd.DataFrame(bastion_vols).to_numpy().T.reshape((1, strikes.shape[1], -1))

    def _build_volatility_surface(val_date, expiry_times, expiries, strikes, iv, dtype):
        interpolator = tff.math.interpolation.interpolation_2d.Interpolation2D(
                expiry_times, strikes, iv, dtype=dtype)

        def _interpolator(t, x):
            x_transposed = tf.transpose(x)
            t = tf.broadcast_to(t, x_transposed.shape)
            return tf.transpose(interpolator.interpolate(t, x_transposed))

        return VolatilitySurface(val_date, expiries, strikes, iv, interpolator=_interpolator, dtype=dtype)

    return _build_volatility_surface(valuation_date, expiry_times, expiries, strikes, input_vols, dtype=tf.float32)

def graph_volsurface(expiry_times, strikes, bastion_vols):
    ''' graph results '''
    # Graph Volsurface
    t = expiry_times.numpy().reshape(-1)
    zz = pd.DataFrame(bastion_vols).to_numpy().T.reshape((-1, len(strikes)))
    xx, yy = np.meshgrid(strikes, t)

    plot3d(xx, yy, zz, set_z=True)

def custom_discount_factor_fn_helper(LOOKUP_forwards, LOOKUP_time, SPOT_REF, t):
    ''' calculates piecewise r from market forwards
        wings -> keep r constant-ish
        inbetween -> np.log(x/y) / (t_x-t_y)
    '''
    # t_query > t_market_data -> 1+ year away
    if t > LOOKUP_time[-1]:
        r = np.log(LOOKUP_forwards[-1]/LOOKUP_forwards[-2])/(LOOKUP_time[-1] - LOOKUP_time[-2])
    # t_query < t_market_data -> 0-2 weeks
    elif t < LOOKUP_time[0]:
        r = np.log(LOOKUP_forwards[0]/SPOT_REF)/LOOKUP_time[0]
    # in range
    else:
        time_diff = t - LOOKUP_time
        # check if t matches a value exactly
        if (time_diff == 0).sum() == 1:
            indx = np.where(time_diff == 0)[0][0]
            r = (np.log(LOOKUP_forwards[indx]/LOOKUP_forwards[indx-1])/(LOOKUP_time[indx]-LOOKUP_time[indx-1]))
        else:
            x = LOOKUP_forwards[np.where(time_diff > 0, time_diff, np.inf).argmin()]
            x_t = LOOKUP_time[np.where(time_diff > 0, time_diff, np.inf).argmin()]
            y = LOOKUP_forwards[np.where(time_diff < 0, time_diff, -np.inf).argmax()]
            y_t = LOOKUP_time[np.where(time_diff < 0, time_diff, -np.inf).argmax()]
            r = np.log(y/x) / (y_t - x_t)
    return r

def precompute_piecewise_r(critical_dates, bastion_forwards, expiry_times, SPOT_REF):
    ''' pre-computes r as a hashmap for tensor compatibility
    '''
    timesteps = np.array(list(range(15000)))
    piecewise_r = []
    # Isolate the forward values that actually come from the market data
    LOOKUP_forwards = np.array([bastion_forwards[m.strftime('%d%b%y').upper()] for m in critical_dates])
    LOOKUP_time = np.array([t for i, t in enumerate(expiry_times.numpy().reshape(-1)) if list(bastion_forwards.keys())[i] in [m.strftime('%d%b%y').upper() for m in critical_dates]])
    for t in timesteps:
        # Compute piecewise r as a function of timestep as fraction of year -> 1/100000
        piecewise_r.append(custom_discount_factor_fn_helper(LOOKUP_forwards, LOOKUP_time, SPOT_REF, t/10000))

    timesteps = tf.convert_to_tensor(timesteps, dtype=tf.int32)
    piecewise_r = tf.convert_to_tensor(np.array(piecewise_r), dtype=tf.float32)
    # Create a lookup hashmap accurate to 1/10000 year timestep
    global R_LOOKUP
    # Index of this hashmap corresponds to 1 -> 150000 where 1 is 1/100000 of a year
    R_LOOKUP = tf.lookup.StaticHashTable(tf.lookup.KeyValueTensorInitializer(timesteps, piecewise_r), default_value=-1)

def check_piecewise_r():
    ''' visualise R_LOOKUP by generating a continuous input simulating time '''
    times = tf.constant(np.linspace(0, 1, 100).reshape(-1,1))
    rates = R_LOOKUP.lookup(tf.cast(times*10000, tf.int32)).numpy()
    plt.plot(times,rates)
    plt.grid()
    plt.show()

def check_local_volsurface(SPOT_REF, volsurface, dtype=tf.float32):
    ''' plot local volsurface '''
    dupire_local_volatility = _dupire_local_volatility_iv

    initial_spot = tf.constant([[SPOT_REF]], dtype=dtype)
    dividend_yield = tf.constant([[0]], dtype=dtype)

    k = 20
    xa = np.array([i/k for i in range(k)])
    ya = np.array([20000 + 1000*i for i in range(31)])
    lvs = np.empty((xa.shape[0], ya.shape[0]))
    # loop through time while strikes are are tensor
    for xxx, xai in enumerate(xa):
        times = tf.convert_to_tensor([xai], dtype=dtype)
        spots = tf.reshape(tf.constant([ya], dtype=dtype), [-1, 1])
        # Compute local vol and safe it
        lvs[xxx] = dupire_local_volatility(times, spots, initial_spot, volsurface.volatility, custom_discount_factor_fn, dividend_yield).numpy().reshape(-1)

    # Graph it
    xx, yy = np.meshgrid(ya, xa)
    plot3d(xx, yy, lvs, set_z=False)

def plot3d(x, y, z, set_z=False):
    ''' 3d plotter '''
    plt.figure(figsize=(10, 9))
    ax = plt.axes(projection='3d')
    # print(len(S), t.shape, xx.shape, yy.shape, zz.shape)
    ax.contourf3D(y, x, z, 300, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    if set_z:
        ax.set_zlim(0.5, 1.5)
    plt.show()

def setup_vanilla_benchmark(valuation_date, critical_dates, SPOT_REF, dtype=tf.float32):
    ''' Defines the input parameters for the vanilla benchmark of the monte carlo'''
    # strikes used for the vanilla closed form solution comparison
    strikes = tf.constant([15000 + 1000*i for i in range(21)], dtype=dtype)
    # dates used for snapshot
    critical_dates = tff.datetime.dates_from_datetimes(critical_dates).reshape((1,-1))
    expiries = tff.datetime.daycount_actual_365_fixed(start_date=valuation_date, end_date=critical_dates, dtype=dtype)
    expiries = expiries.numpy()[0].tolist()
    return strikes, expiries

def get_blackscholes_vols(strikes, dates, market_vols, market_forwards):
    ''' Extract the IVs from the initial market data vol dataframe to later use it for BS option pricing '''
    bs_extracted_ivs = np.empty((strikes.shape[0], len(dates)))
    forwards = np.empty(len(dates))

    volsurface_refs = pd.DataFrame(market_vols)

    for a, m in enumerate(dates):
        m = m.strftime('%d%b%y').upper()
        forwards[a] = market_forwards[m]
        for b, k in enumerate(strikes):
            k = k.numpy()
            bs_extracted_ivs[b, a] = volsurface_refs.loc[str(k), m]

    return bs_extracted_ivs, forwards

def compute_blackscholes_prices(market_vols, market_forwards, expiries, strikes, SPOT_REF, RISK_FREE_RATE=0):
    ''' compute blackscholes option prices from data '''

    bs_computed_prices = np.empty(market_vols.shape)

    # Use for separation of calls and puts
    # if strike > 30000 upside OTM calls
    is_call_option = np.array([strike > 22000 for strike in strikes])

    USE_RISK_FREE_RATE = False # np.log(Ft/S0)/t when t in years

    # Use the market forward data to convert between prices
    if not USE_RISK_FREE_RATE:
        # Loop through expiries and computes for all strikes
        for a in range(bs_computed_prices.shape[1]):
            v = market_vols[:, a]
            bs_computed_prices[:,a] = option_price(
                                    volatilities=v,
                                    strikes=strikes.numpy()/ market_forwards[a],
                                    expiries=np.array([expiries[a]]*len(strikes)),
                                    forwards=1,
                                    discount_factors=1,
                                    is_call_options=is_call_option
                                    ).numpy() * market_forwards[a]
    # Use the extrapolated Forward risk free rate from the futures
    else:
        for a in range(bs_computed_prices.shape[1]):
            v = market_vols[:, a]
            bs_computed_prices[:,a] = option_price(
                                    volatilities=v,
                                    strikes=strikes.numpy()/ (SPOT_REF * np.exp(RISK_FREE_RATE * expiries[a])) ,
                                    expiries=np.array([expiries[a]]*len(strikes)),
                                    forwards=1,
                                    discount_factors=1,
                                    is_call_options=is_call_option
                                    ).numpy() * (SPOT_REF * np.exp(RISK_FREE_RATE * expiries[a]))
    return bs_computed_prices

def run_montecarlo(expiries, volsurface, SPOT_REF, num_samples, num_batches, dt, dtype=tf.float32):
    ''' main function to run montecarlo and collect the results '''
    montecarlo_function = tf.function(setup_pricer(expiries, volsurface, num_samples, dt, dtype))#, jit_compile=True)
    spot = tf.constant(SPOT_REF, dtype=dtype)
    start = time.time()
    print("START MC")
    all_paths = []
    # Run in multiple batches
    for i in range(num_batches):
        paths = montecarlo_function(spot)
        all_paths.append(paths)
        end = time.time()
        print(f"END ITER. No. {i+1}/{num_batches}")
        print(f"Duration {i+1}/{num_batches}: {end-start:.1f} s")
        start = time.time()
    # Collect and return the results in (samples x expiries) numpy array
    path_collection = None
    for k, p in enumerate(all_paths):
        if k==0:
            path_collection = p.numpy().reshape(p.shape[0], -1)
        else:
            path_collection = np.concatenate((path_collection, p.numpy().reshape(p.shape[0], -1)))
    print(path_collection.shape)
    for i in range(path_collection.shape[1]):
        print(f"Standard error \
            {tf.nn.relu(20000 - path_collection[:, i]).numpy().std()/np.sqrt(path_collection[:, i].shape[0])/SPOT_REF*10000:.2f} bps \
            at price \
            {tf.nn.relu(20000 - path_collection[:, i]).numpy().mean()/SPOT_REF*100:.4f}%")
    return path_collection

def setup_pricer(expiries, volsurface, num_samples, dt, dtype=tf.float32):
    """ Set up option pricing function """
    def simulate_itoprocess(spot):
        ''' Use LocalVolatilityModel class to set up the Ito process:
            - passes the volsurface and spot etc using dupire from iv's
            - then calls __init__() for initialise localvolmodel '''
        # HAS to be spot and not Log spot -> converted in _dupire_local_volatility_iv()
        process = LocalVolatilityModel.from_volatility_surface(dim = 1,
                                                spot = spot,
                                                implied_volatility_surface = volsurface,
                                                discount_factor_fn = custom_discount_factor_fn, #lambda t: tf.math.exp(-RISK_FREE_RATE* t),
                                                dtype=dtype)
        # HAS to be spot -> is converted to log spot in sample_paths()
        paths = process.sample_paths(
            expiries,
            num_samples=num_samples,
            initial_state=spot,
            random_type=tff.math.random.RandomType.PSEUDO_ANTITHETIC,
            time_step=dt)
        return paths
    return simulate_itoprocess

@tf.function
def custom_discount_factor_fn(t):
    ''' custom discount factor fn using piecewise r as a function of a tensor t'''
    indx = tf.cast(t*10000, tf.int32)
    rates = R_LOOKUP.lookup(indx)
    return tf.math.exp(-rates*t)

def main(today_date, reference_date, critical_dates, graph_surface=True, graph_piecewise_r=True, graph_localvol=True):
    ''' runs everything  '''
    # MonteCarlo parameters
    num_samples = tf.constant(60000, dtype=tf.int32)
    num_timesteps = tf.constant(300, dtype=tf.float32)
    num_batches = tf.constant(10,  dtype=tf.int64)
    dt = tf.constant(1. / num_timesteps,  dtype=tf.float32)

    # Gets Data and constructs volsurface
    valuation_date, surface_dates, surface_expiries, surface_expiry_times, surface_strikes = generate_input_data(today_date, reference_date)
    bastion_vols, bastion_forwards, SPOT_REF = get_bastion_data(surface_dates, surface_strikes)
    volsurface = construct_volsurface(valuation_date, surface_expiries, surface_expiry_times, surface_strikes, bastion_vols)
    precompute_piecewise_r(critical_dates, bastion_forwards, surface_expiry_times, SPOT_REF)

    # Run visual checks where desired
    if graph_surface:
        graph_volsurface(surface_expiry_times, surface_strikes, bastion_vols)
    if graph_piecewise_r:
        check_piecewise_r()
    if graph_localvol:
        check_local_volsurface(SPOT_REF, volsurface)

    # Set up the vanilla comparison & compute the vanilla option components
    bs_strikes, bs_expiries = setup_vanilla_benchmark(today_date, critical_dates, SPOT_REF)
    bs_vols, market_forwards = get_blackscholes_vols(bs_strikes, critical_dates, bastion_vols, bastion_forwards)
    bs_computed_prices = compute_blackscholes_prices(bs_vols, market_forwards, bs_expiries, bs_strikes, SPOT_REF)

    # Run Montecarlo
    bs_expiries = tf.convert_to_tensor(bs_expiries, tf.float32)
    run_montecarlo(bs_expiries, volsurface, SPOT_REF, num_samples, num_batches, dt)


if __name__ == '__main__':

    TODAY = datetime.date(2022, 6, 21)
    REF_DAY = datetime.date(2022, 7, 6)

    CRITICAL_DATES = [
    datetime.date(2022, 7, 8),
    datetime.date(2022, 7, 29),
    datetime.date(2022, 8, 26),
    datetime.date(2022, 9, 30),
    datetime.date(2022, 12, 30),
    datetime.date(2023, 3, 31)
    ]

    main(TODAY, REF_DAY, CRITICAL_DATES)
