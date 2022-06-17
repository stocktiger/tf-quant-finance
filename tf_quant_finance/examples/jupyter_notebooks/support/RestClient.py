"""
    This module implements the Local Client to the BASTION volsurface
"""

import time
import urllib.parse
from typing import Optional, Dict, Any

import hmac
import json
import requests

class RestClient:
    def __init__(self, api_url, api_key=None, api_secret=None, subaccount_name=None) -> None:
        self.API_URL = api_url
        self._api_key = api_key
        self._api_secret = api_secret
        self._subaccount_name = subaccount_name

    def _get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('GET', path, params=params)

    def _post(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('POST', path, payload=params)

    def _delete(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request('DELETE', path, payload=params)

    def _request(self, method: str, endpoint: str, auth: bool = True, params={}, payload={}, api=None):
        for count in range(1,4):
            try:
                _payload = json.dumps(payload) if payload else ''
                _endpoint=f'{endpoint}?{urllib.parse.urlencode(params, safe="/")}' if params else _endpoint
                if not api:
                    api = self.API_URL

                url = f'{api}{_endpoint}'

                header = {}
                if method == 'GET':
                    response = requests.get(url, headers=header, timeout=3)
                elif method == 'POST':
                    response = requests.post(url, data=_payload, headers=header, timeout=3)
                elif method == 'DELETE':
                    response = requests.delete(url, data=_payload, headers=header, timeout=3)

                data = response.json()
                response.close()
                return data
            except ConnectionError as err:
                time.sleep(0.5*count)
            except Exception as err:
                time.sleep(0.5*count)

    def _generate_signature(self, timestamp: str, method: str, endpoint: str, data='') -> dict:
        message = f'{timestamp}{method}{endpoint}{data}'
        signature = hmac.new(self._api_secret.encode('utf-8'), message.encode('utf-8'), 'sha256').hexdigest()
        return signature

    def get_endpoint(self, endpoint, params={}):
        return self._get(endpoint, params)

    def post_endpoint(self, endpoint, params={}):
        return self._post(endpoint, params)

