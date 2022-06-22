import datetime
import numpy as np
import pandas as pd

import tensorflow as tf
import tf_quant_finance as tff 
from tf_quant_finance.experimental.pricing_platform.framework.market_data.volatility_surface import VolatilitySurface
from tf_quant_finance.experimental.local_volatility import local_volatility_model

@tf.function
def dupire_local_volatility(*arg, **kwargs):
    return local_volatility_model._dupire_local_volatility_iv(*arg, **kwargs)

from vol_surface_client import client
from utils import query_spot_price

def build_volatility_surface(val_date, expiry_times, expiries, strikes, iv, dtype):
  interpolator = tff.math.interpolation.interpolation_2d.Interpolation2D(expiry_times, strikes, iv, dtype=dtype)
  
  def _interpolator(t, x):
    x_transposed = tf.transpose(x)
    t = tf.broadcast_to(t, x_transposed.shape)
    return tf.transpose(interpolator.interpolate(t, x_transposed))

  return VolatilitySurface(val_date, expiries, strikes, iv, interpolator=_interpolator, dtype=dtype)

def get_vol_surface_bastion(S):
    dtype = tf.float32
    # 1. Feed Volatility surface using a processed_market_data.VolatilitySurface instance
    DAY_REF = datetime.date.today() #YYYY-MM-DD
    # DAY_REF = datetime.date(2022, 6, 8) #YYYY-MM-DD

    dates = [DAY_REF + datetime.timedelta(days=i) for i in range(30*6+15, 30*6+16)]
    
    # SET expiries as a (1 x Maturities) tensor 
    expiries = tff.datetime.dates_from_datetimes(dates).reshape((1,-1))

    # Construct the strike space
    dk = 100
    # S = [15000 + dk*i for i in range(45000//dk)]
    
    # New addition
    # SET strikes as (1 x Maturities x Strikes) tensor
    strikes = np.array(S)
    strikes = [expiries.shape[1]*[strikes.tolist()]]
    strikes = tf.constant(strikes, dtype=dtype)

    # Obtain IVs from volsurface
    date_strings = ",".join([date.strftime('%d%b%y').upper() for date in dates])
    strikes_strings = ",".join([str(round(strike,4)) for strike in S])
    params =  {"symbols": "BTC", "maturities": date_strings, "strikes": strikes_strings}

    # dict: { maturity: {price: iv, ... }, ... }
    response = client.get_endpoint('volsurface', params)
    return pd.DataFrame(response).to_numpy().T.squeeze()

def get_tf_volsurface_from_bastion(dtype=tf.float64):
    # 1. Feed Volatility surface using a processed_market_data.VolatilitySurface instance
    DAY_REF = datetime.date.today() #YYYY-MM-DD
    # DAY_REF = datetime.date(2022, 6, 8) #YYYY-MM-DD

    # Construct tff native Datetensors
    valuation_date = tff.datetime.dates_from_datetimes([DAY_REF])
    dates = [DAY_REF + datetime.timedelta(days=i) for i in range(1, 30*6+15)]

    # SET expiries as a (1 x Maturities) tensor 
    expiries = tff.datetime.dates_from_datetimes(dates).reshape((1,-1))

    # Construct the strike space
    dk = 100
    S = [15000 + dk*i for i in range(45000//dk)]

    # Obtain IVs from volsurface
    date_strings = ",".join([date.strftime('%d%b%y').upper() for date in dates])
    strikes_strings = ",".join([str(strike) for strike in S])
    params =  {"symbols": "BTC", "maturities": date_strings, "strikes": strikes_strings}

    # dict: { maturity: {price: iv, ... }, ... }
    response = client.get_endpoint('volsurface', params)

    # New addition
    # SET strikes as (1 x Maturities x Strikes) tensor
    strikes = np.array(S)
    strikes = [expiries.shape[1]*[strikes.tolist()]]
    strikes = tf.constant(strikes, dtype=dtype)
    #strikes = tf.math.log(strikes)

    input_vols = tf.convert_to_tensor(pd.DataFrame(response).to_numpy().T.reshape((1, strikes.shape[1], -1)), dtype=dtype)

    expiry_times = tff.datetime.daycount_actual_365_fixed(start_date=valuation_date, end_date=expiries, dtype=dtype)


    # Test that it returns the right surface
    volsurface = build_volatility_surface(valuation_date, expiry_times, expiries, strikes, input_vols, dtype=dtype)
    return volsurface

if __name__ == "__main__":
    
    SPOT_REF = query_spot_price("BTC")
    volsurface = get_tf_volsurface_from_bastion()

    # Test that it works with dates
    print(volsurface.volatility(strike=np.array([[40000]]), 
                                expiry_dates=tff.datetime.dates_from_datetimes([datetime.date(2022, 9, 1)]).reshape((1,-1))))

    # Test that it works with time
    print(volsurface.volatility(strike=np.array([[32100]]), 
                                expiry_times=np.array([[1/12]])))
    print("Lower limit:", volsurface.volatility(strike=np.array([[0]]), 
                                expiry_times=np.array([[1/12]])).numpy())
    print("Upper limit:", volsurface.volatility(strike=np.array([[100000]]), 
                                expiry_times=np.array([[1/12]])).numpy())


    # Query specific times and spots 
    # if times.shape > spots.shape => ERROR (?)
    times = tf.convert_to_tensor([1/12], dtype=tf.float64)
    spots = tf.reshape(tf.convert_to_tensor([30000, 32000, 33000], dtype=tf.float64), [-1, 1])

    # Should remain constant
    initial_spot = tf.constant([[SPOT_REF]], dtype=tf.float64)
    discount_factor_fn = lambda t: tf.math.exp(-0 * t)
    # dividend_yield = tf.convert_to_tensor([0.], dtype=tf.float64)
    dividend_yield = [0.]

    # Looks like all the spot/strike inputs in non-log terms
    # -> Double checks 
    # Query function
    local_vol = dupire_local_volatility(times, 
                                        spots, 
                                        initial_spot, 
                                        volsurface.volatility, 
                                        discount_factor_fn, 
                                        dividend_yield)

    print(local_vol)