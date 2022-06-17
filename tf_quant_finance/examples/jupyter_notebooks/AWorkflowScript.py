'''
    Contains the MC simulation implementation using Bastion Vol
'''

import datetime

# Import all the libraries as required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_quant_finance as tff
from tf_quant_finance.black_scholes import option_price
from tf_quant_finance.experimental.pricing_platform.framework.market_data.volatility_surface import \
    VolatilitySurface

from support.LocalVolModel import LocalVolatilityModel, _dupire_local_volatility_iv
from support.RestClient import RestClient

CLIENT = RestClient("https://defi-bot.bastioncb.com:1443/")

def generate_input_data(today_date, reference_date):
    ''' puts everything into tensors where necessary'''
    # Specify date
    valuation_date = tff.datetime.dates_from_datetimes([today_date])
    # Query approximately 1 year of volsurface data on a daily interval
    dates = [reference_date +
                     datetime.timedelta(days=i) for i in range(1, 30*12)]
    # SET expiries as a (1 x Maturities) tensor
    expiries = tff.datetime.dates_from_datetimes(dates).reshape((1, -1))
    expiry_times = tff.datetime.daycount_actual_365_fixed(
            start_date=valuation_date, end_date=expiries, dtype=tf.float64)

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

def construct_volsurface(valuation_date, expiries, expiry_times, strikes, bastion_vols):
    ''' construct volsurface from input data and return VolatilitySurface object'''

    # SET strikes as (1 x Maturities x Strikes) tensor
    strikes = [expiries.shape[1]*[strikes]]
    strikes = tf.constant(strikes, dtype=tf.float64)

    input_vols = pd.DataFrame(bastion_vols).to_numpy().T.reshape((1, strikes.shape[1], -1))

    def _build_volatility_surface(val_date, expiry_times, expiries, strikes, iv, dtype):
        interpolator = tff.math.interpolation.interpolation_2d.Interpolation2D(
                expiry_times, strikes, iv, dtype=tf.float64)

        def _interpolator(t, x):
            x_transposed = tf.transpose(x)
            t = tf.broadcast_to(t, x_transposed.shape)
            return tf.transpose(interpolator.interpolate(t, x_transposed))

        return VolatilitySurface(val_date, expiries, strikes, iv, interpolator=_interpolator, dtype=dtype)

    return _build_volatility_surface(valuation_date, expiry_times, expiries, strikes, input_vols, dtype=tf.float64)

def graph_volsurface(expiry_times, strikes, bastion_vols):
    ''' graph results '''
    # Graph Volsurface
    t = expiry_times.numpy().reshape(-1)
    zz = pd.DataFrame(bastion_vols).to_numpy().T.reshape((-1, len(strikes)))
    xx, yy = np.meshgrid(strikes, t)

    plot3d(xx, yy, zz, set_z=True)

def custom_discount_factor_fn_helper(bastion_forwards, expiry_times, SPOT_REF, t):
    ''' calculates piecewise r
        wings -> keep r constant-ish
        inbetween -> np.log(x/y) / (t_x-t_y)

    '''
    # Check if t is out of range
    LOOKUP_forwards = np.array(list(bastion_forwards.values()))
    LOOKUP_time = expiry_times.numpy().reshape(-1)

    if t > LOOKUP_time[-1]:
        r = np.log(LOOKUP_forwards[-1]/LOOKUP_forwards[-2])/(LOOKUP_time[-1] - LOOKUP_time[-2])

    # Check if t is out of range
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

def precompute_piecewise_r(bastion_forwards, expiry_times, SPOT_REF):
    ''' pre-computes r as a hashmap for tensor compatibility
    '''
    timesteps = np.array(list(range(15000)))
    piecewise_r = []
    for t in timesteps:
        # Compute piecewise r as a function of timestep as fraction of year -> 1/100000
        piecewise_r.append(custom_discount_factor_fn_helper(bastion_forwards, expiry_times, SPOT_REF, t/10000))

    timesteps = tf.convert_to_tensor(timesteps)
    piecewise_r = tf.convert_to_tensor(np.array(piecewise_r))
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

def check_local_volsurface(SPOT_REF, volsurface):
    ''' plot local volsurface '''
    dupire_local_volatility = _dupire_local_volatility_iv

    initial_spot = tf.constant([[SPOT_REF]], dtype=tf.float64)
    dividend_yield = tf.constant([[0]], dtype=tf.float64)

    k = 20
    xa = np.array([i/k for i in range(k)])
    ya = np.array([20000 + 1000*i for i in range(31)])
    lvs = np.empty((xa.shape[0], ya.shape[0]))
    # loop through time while strikes are are tensor
    for xxx, xai in enumerate(xa):
        times = tf.convert_to_tensor([xai], dtype=tf.float64)
        spots = tf.reshape(tf.constant([ya], dtype=tf.float64), [-1, 1])
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

@tf.function
def custom_discount_factor_fn(t):
    ''' custom discount factor fn using piecewise r as a function of a tensor t'''
    #
    indx = tf.cast(t*10000, tf.int32)
    rates = R_LOOKUP.lookup(indx)
    return tf.math.exp(-rates*t)


def main(today_date, reference_date, critical_dates, graph_surface=False, graph_piecewise_r=False, graph_localvol=False):
    ''' runs everything  '''

    # Gets Data and constructs volsurface
    valuation_date, dates, expiries, expiry_times, strikes = generate_input_data( today_date, reference_date)
    bastion_vols, bastion_forwards, SPOT_REF = get_bastion_data(dates, strikes)
    volsurface = construct_volsurface(valuation_date, expiries, expiry_times, strikes, bastion_vols)
    precompute_piecewise_r(bastion_forwards, expiry_times, SPOT_REF)

    # Run visual checks where desired
    if graph_surface:
        graph_volsurface(expiry_times, strikes, bastion_vols)
    if graph_piecewise_r:
        check_piecewise_r()
    if graph_localvol:
        check_local_volsurface(SPOT_REF, volsurface)

    # Set up the montecarlo and vanilla comparison


if __name__ == '__main__':

    TODAY = datetime.date(2022, 6, 17)
    REF_DAY = datetime.date(2022, 7, 2)

    CRITICAL_DATES = [
    datetime.date(2022, 7, 1),
    datetime.date(2022, 7, 8),
    datetime.date(2022, 7, 29),
    datetime.date(2022, 8, 26),
    datetime.date(2022, 9, 30),
    datetime.date(2022, 12, 30),
    datetime.date(2023, 3, 31)
    ]

    main(TODAY, REF_DAY, CRITICAL_DATES)
