import json
from requests import Session


def query_spot_price(coin_symbol: str):
    ''' Current API limit at 333 calls per day so don't go crazy please '''

    url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest'
    parameters = {'symbol': coin_symbol}
    headers = {'Accepts': 'application/json',
            'X-CMC_PRO_API_KEY': 'de5e3133-b7e5-41c2-a01d-116a71d62aa4', }

    session = Session()
    session.headers.update(headers)
    try:
        response = session.get(url, params=parameters)
        data = json.loads(response.text)

        if coin_symbol in data["data"].keys():
            price = data["data"][coin_symbol]["quote"]["USD"]["price"]
        elif coin_symbol.upper() in data["data"].keys():
            price = data["data"][coin_symbol.upper()]["quote"]["USD"]["price"]
        print(f"Current {coin_symbol} price: {price:,.2f} USD")

    except Exception as e:
        print(e)
    return float(price)
