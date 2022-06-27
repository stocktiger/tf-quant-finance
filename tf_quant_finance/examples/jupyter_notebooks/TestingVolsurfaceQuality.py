'''
    Tests the volsurfaces w.r.t
        - fitted volsurface model
        - market data
        - tf model
'''
import asyncio
import datetime
import time
import urllib

import aiohttp
# Import all the libraries as required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tf_quant_finance as tff
# Get available instruments from tardis
from tardis_client import Channel, TardisClient
from tf_quant_finance.experimental.pricing_platform.framework.market_data.volatility_surface import \
    VolatilitySurface

from support.RestClient import RestClient

CLIENT = RestClient("https://defi-bot.bastioncb.com:1443/")

START = datetime.datetime.utcfromtimestamp(int(time.time())-60*30).strftime('%Y-%m-%dT%H:%M:%S')
END = datetime.datetime.utcfromtimestamp(int(time.time())-60*15).strftime('%Y-%m-%dT%H:%M:%S')

async def get_instruments(start, end):
    ''' gets all instruments avaiable at the specific timestamp
    Note: as (end - start) is usually 15 mins it'll get only the instruments that are being updated in this timeframe.
    From testing usually over 95-99% of the instruments are saved but not a guaranteed 100%. All instruments of interest should be saved.
    '''
    tardis_client = TardisClient(api_key="TD.L9M2r84pFJlqi-Te.vbmdgRxOh7mmRW1.yvv8sfStcf1bF84.Di8MM3XFos4z6oP.UQtLGISKg2zwOob.X40S")
    print("start tardis")
    messages = tardis_client.replay(
        exchange="deribit",
        from_date=start,
        to_date=end,
        filters=[
            Channel(name="markprice.options", symbols=[]),
        ],
    )
    options = set()
    async for local_timestamp, message in messages:
        datas = message["params"]["data"]
        for data in datas:
            c, m, s, t = data["instrument_name"].split('-')
            if (c == 'BTC'):
                options.add(f'{m}-{s}')
    print('options found in first 10 mins:', len(options))
    return options

def generate_input_data(today_date, reference_date, dtype=tf.float32):
    ''' puts everything into tensors where necessary'''
    # Specify date
    valuation_date = tff.datetime.dates_from_datetimes([today_date])
    # Query approximately 1 year of volsurface data on a daily interval
    dates = [reference_date + datetime.timedelta(days=i) for i in range(1, 30*12)]
    # SET expiries as a (1 x Maturities) tensor
    expiries = tff.datetime.dates_from_datetimes(dates).reshape((1, -1))
    expiry_times = tff.datetime.daycount_actual_365_fixed(start_date=valuation_date, end_date=expiries, dtype=dtype)
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
    # volsurface dictionary
    response = CLIENT.get_endpoint('volsurface/forward_price', params)
    print(response.keys())
    response_ivs, response_forwards, SPOT_REF = response["iv"], response["forward_price"], response["spot"]

    print("SPOT REFERENCE", SPOT_REF)
    return response_ivs, response_forwards, SPOT_REF

async def get_market_data(available_instruments):
    ''' guarantees same cycle '''
    available_instruments = {inst for inst in available_instruments if inst.split('-')[0]!='22JUN22'}
    complete = False
    while not complete:
        print('trial')
        forwards = {}
        vols = []
        spot = 0
        first = True
        REF_TIME = 0
        async with aiohttp.ClientSession() as session:
            for i, inst in enumerate(available_instruments):
                m, s = inst.split('-')
                params = {"symbols": "BTC", "maturities": m, "strikes": s}
                async with session.get(f"https://defi-bot.bastioncb.com:1443/fittedVol/single?{urllib.parse.urlencode(params, safe='/')}", headers={}, timeout=3) as resp:
                    data = await resp.json()
                    if first:
                        spot = data["ref_sym_spot"]
                        REF_TIME = data["last_update_ts"]
                        first = False

                    if REF_TIME != data["last_update_ts"]:
                        print('ABORT: Restart Cell')
                        break

                    if m not in forwards.keys():
                        forwards[m] = data["forward_price"]
                    vols.append((m, s, data["volBTCDeribit"]))
                if i +1 == len(available_instruments):
                    complete = True
                    print("Done successfully")
    return forwards, vols, spot

def modify_market_vols(v, f):
    # Fix market data -> Check if output is okay
    MATS_TO_GET_RID_OF_FRONT = 0
    STRIKES_REQUIREMENT_MISSING = 2

    # Get all strikes from market data
    all_strikes = []
    for item in v:
        _, strk, _ = item
        if int(strk) not in all_strikes:
            all_strikes.append(int(strk))

    # Sort strikes and maturities
    all_strikes = np.sort(np.array(all_strikes))
    all_maturities = np.array(sorted(list(f.keys()), key=lambda x: datetime.datetime.strptime(x, '%d%b%y')))
    print('All_mats', all_maturities)
    # Create 2D market data volsurface
    matrix_vol = np.zeros((1, len(all_maturities), len(all_strikes)))
    for item in v:
        mat, strk, ivx = item
        matrix_vol[0, np.where(mat == all_maturities)[0][0], np.where(int(strk) == all_strikes)[0][0]] = ivx

    # Check the matrix to get rid of sparse entries (i.e. maturities and strikes)
    # Get rid of usually 1-d and 2-d maturities -> Keep Fridays
    keep_strikes = []
    # Loop through strikes to identify good and bad strikes
    for i in range(matrix_vol.shape[2]):
        if (matrix_vol[0, MATS_TO_GET_RID_OF_FRONT:, i] == 0).sum() <=STRIKES_REQUIREMENT_MISSING:
            keep_strikes.append(all_strikes[i])
    keep_strikes = np.array(keep_strikes)
    #keep_strikes = keep_strikes[keep_strikes<=60000]

    _, x_indx, _ = np.intersect1d(all_strikes, keep_strikes, return_indices=True)

    # Reset the matrix to filtered data
    all_strikes = all_strikes[x_indx]
    all_maturities = all_maturities[MATS_TO_GET_RID_OF_FRONT:]
    matrix_vol = matrix_vol[:, MATS_TO_GET_RID_OF_FRONT:, x_indx]
    print('All_mats', all_maturities)

    # Loop through maturities to fill in strikes
    for j in range(matrix_vol.shape[1]):
        vols_per_mat = matrix_vol[0, j, :]
        start = True
        for i, vn in enumerate(vols_per_mat):
            # make sure it's not the last entry
            if i < all_strikes.shape[0]:
            # if first item is non zero
                if (vols_per_mat[i] != 0) and start:
                    start = False
                # if first item is zero
                if (vn == 0) and start:
                    i_temp = 0
                    while (vols_per_mat[i_temp] == 0):
                        i_temp += 1
                        # if next item is also zero
                        if (vols_per_mat[i_temp] == 0):
                            continue
                        # once a non-zero item is found -> fill from right to left
                        else:
                            i_val = i_temp
                            while i_temp > 0:
                                i_temp -= 1
                                matrix_vol[0, j, i_temp] = vols_per_mat[i_val]
                                start = False
                # at first zero value after start
                elif (vn == 0):
                    # check if it's the end -> fill from left to right
                    if vols_per_mat[i:].sum() == 0:
                        i_temp = i
                        while i_temp < all_strikes.shape[0]:
                            matrix_vol[0, j, i_temp] = vols_per_mat[i-1]
                            i_temp += 1
                    # linearly interpolate the rest
                    else:
                        i_temp, i_start = i, i-1
                        # see how long the zeros range
                        while (vols_per_mat[i_temp] == 0):
                            i_temp += 1
                            if vols_per_mat[i_temp] > 0:
                                i_end = i_temp
                        matrix_vol[0, j, i_start+1:i_end] = np.interp(all_strikes[i_start+1:i_end],[all_strikes[i_start], all_strikes[i_end]],[vols_per_mat[i_start], vols_per_mat[i_end]])

    all_maturities = np.array([datetime.datetime.strptime(mat, '%d%b%y').strftime('%d%b%y').upper() for mat in all_maturities])
    return all_strikes, all_maturities, matrix_vol[0]

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

def reformat_bastion_data(bastion_vols, MKT_maturities, MKT_strikes, MKT_vols):
    ''' surface is bounded by the maturities and strikes'''
    bastion_surface = np.zeros(MKT_vols.shape)
    for m_i, m in enumerate(MKT_maturities):
        for k_i, k in enumerate(MKT_strikes):
            bastion_surface[m_i, k_i] = bastion_vols[m][f"{k}.0"]
    return bastion_surface

def plot_surfaces_full(volsurface, surface1, today_date, MKT_maturities, MKT_strikes):
    ''' used for the volsurface plot across the whole strike space'''
    critical_dates = [datetime.datetime.strptime(mat, '%d%b%y') for mat in  MKT_maturities]
    critical_dates = tff.datetime.dates_from_datetimes(critical_dates).reshape((1,-1))
    t = tff.datetime.daycount_actual_365_fixed(start_date=today_date, end_date=critical_dates)
    t = t.numpy()[0]
    # Get volsurface response which should be the same as fitted BASTION_vol
    surface3 = np.zeros(surface1.shape)
    for m_i, m in enumerate(t):
        for k_i, k in enumerate(MKT_strikes):
            surface3[m_i, k_i] = volsurface.volatility(strike=[[k]], expiry_times=[[m]])

    xx, yy = np.meshgrid(MKT_strikes, t)
    print(surface1)
    plt.figure(figsize=(10, 9))
    ax = plt.axes(projection='3d')
    ax.contourf3D(yy, xx, surface3, 300, cmap='rainbow')
    ax.contourf3D(yy, xx, surface1, 300, cmap='ocean')
    #ax.contourf3D(yy, xx, surface1-surface3, 300, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_zlim(0.5, 1.5)
    plt.show()

def plot_surfaces_diff(surface1, surface2, today_date, MKT_maturities, MKT_strikes):
    ''' used to plot only in the range that the model vol is queried > 2 weeks or REF_date'''
    critical_dates = [datetime.datetime.strptime(mat, '%d%b%y') for mat in  MKT_maturities]
    critical_dates = tff.datetime.dates_from_datetimes(critical_dates).reshape((1,-1))
    t = tff.datetime.daycount_actual_365_fixed(start_date=today_date, end_date=critical_dates)
    t = t.numpy()[0]
    # Get volsurface response which should be the same as fitted BASTION_vol
    xx, yy = np.meshgrid(MKT_strikes, t)
    plt.figure(figsize=(10, 9))
    ax = plt.axes(projection='3d')
    ax.contourf3D(yy, xx, surface1, 300, cmap='rainbow')
    ax.contourf3D(yy, xx, surface2, 300, cmap='ocean')
    #ax.contourf3D(yy, xx, surface1-surface2, 300, cmap='rainbow')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    #ax.set_zlim(0.5, 1.5)
    plt.show()

def main(today_date, reference_date, compare_market_to_model=False):
    ''' runs everything  '''
    # Gets Data and constructs volsurface
    valuation_date, surface_dates, surface_expiries, surface_expiry_times, surface_strikes = generate_input_data(today_date, reference_date)

    # Get market data
    available_instruments = asyncio.run(get_instruments(START, END))
    MKT_forwards, MKT_vols, MKT_spot = asyncio.run(get_market_data(available_instruments))

    BASTION_vols, bastion_forwards, SPOT_REF = get_bastion_data(surface_dates, surface_strikes)
    print()
    print("Check", SPOT_REF, MKT_spot)
    print()
    # -> Takes the market vol data and makes a consitent matrix of the vol surface in the available ranges
    # Discards
    MKT_strikes, MKT_maturities, MKT_vols = modify_market_vols(MKT_vols, MKT_forwards)

    if compare_market_to_model:
        # -> Only works when reference data > market data
        BASTION_vols_reduced = None #reformat_bastion_data(BASTION_vols, MKT_maturities, MKT_strikes, MKT_vols)
        plot_surfaces_diff(MKT_vols, BASTION_vols_reduced, today_date, MKT_maturities, MKT_strikes)
    else:
        volsurface = construct_volsurface(valuation_date, surface_expiries, surface_expiry_times, surface_strikes, BASTION_vols)
        plot_surfaces_full(volsurface, MKT_vols, today_date, MKT_maturities, MKT_strikes)


if __name__ == '__main__':

    TODAY = datetime.date(2022, 6, 27)
    REF_DAY = datetime.date(2022, 7, 11)

    # here overlaps with the market data of vanillas
    CRITICAL_DATES = [
    datetime.date(2022, 7, 1),
    datetime.date(2022, 7, 8),
    datetime.date(2022, 7, 29),
    datetime.date(2022, 8, 26),
    datetime.date(2022, 9, 30),
    datetime.date(2022, 12, 30),
    datetime.date(2023, 3, 31)
    ]

    main(TODAY, REF_DAY)
