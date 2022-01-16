import json
import os
import yfinance as yf
import torch
import numpy as np

from observer.tickers import tickers5_USD


class CryptoHistory:
    def __init__(self, ticker_name):
        tmp = yf.Ticker(ticker_name)
        self.df = tmp.history(period="max")

        self.validate()

        self.start_date = self.get_start_date()
        self.last_date = self.get_last_date()

    def validate(self):
        assert len(self.df.index.values) > 0

    def get_last_date(self):
        return self.df.index.values[-1]

    def get_start_date(self):
        return self.df.index.values[0]


def compute_time_difference(datetime: np.datetime64):
    x = np.timedelta64(datetime, 'ns')
    days = x.astype('timedelta64[D]')
    return days / np.timedelta64(1, 'D')


def build_obs(start_date_global, last_date_global, observed_entries_list, tickers_list=tickers5_USD):
    # Get currencies with start date posterior to start_date_global
    filtered = []
    obs_dim = int(compute_time_difference(last_date_global - start_date_global))
    for ticker_name in tickers_list:
        data = CryptoHistory(ticker_name)
        if data.start_date <= start_date_global and data.last_date >= last_date_global:
            filtered.append(ticker_name)
            dates_mask = [start_date_global <= date <= last_date_global for date in data.df.index.values]

            tmp_obs = []
            for entry in observed_entries_list:
                tmp_obs += list(data.df[dates_mask][entry])
            tmp_obs = torch.reshape(torch.Tensor(tmp_obs), (1, -1))

            obs = tmp_obs if "obs" not in locals() else torch.cat((obs, tmp_obs), 0)

    print(f"List of observed cryptocurrencies: \n"
          f"{filtered} \n"
          f"Observation shape: {obs.size()}")

    return obs


def save_obs_json(obs_tensor, _json_path):
    obs = obs_tensor.tolist()
    obs_dict = {"obs": obs}
    with open(_json_path, "w") as f:
        json.dump(obs_dict, f, indent=4)


def get_obs_from_json(json_dump):
    with open(json_dump) as f:
        obs_dict = json.load(f)
    return torch.Tensor(obs_dict["obs"])


if __name__ == "__main__":
    # Build observation
    msft = yf.Ticker("ETH-USD")
    data = msft.history(period="max")
    start_date_global, last_date_global = data.index.values[0], data.index.values[-1]
    observed_entries_list = ["Close"]
    obs = build_obs(start_date_global, last_date_global, observed_entries_list)

    # Save it
    json_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "obs.json")
    save_obs_json(obs, json_path)



