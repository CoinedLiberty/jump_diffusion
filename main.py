import requests
import numpy as np
import os
os.environ['MPLBACKEND'] = 'TkAgg'  # or any other backend you prefer
import matplotlib.pyplot as plt
import pandas as pd
import random
from datetime import datetime

def get_data(market, interval, num_call):
    url = "https://api.binance.com/api/v3/klines?symbol="
    url += market
    url += "&interval=" + interval
    url += "&limit=1000"

    data = []

    for i in range(num_call):
        if i == 0:
            response = requests.get(url)
            temp = response.json()
            data += [[x[0], x[4]] for x in temp]
        else:
            if interval == '1s':
                new_url = url + "&endTime=" + str(temp[0][0]-1000)
            elif interval == '1m':
                new_url = url + "&endTime=" + str(temp[0][0]-60000)
            response = requests.get(new_url)
            temp = response.json()
            data += [[x[0], x[4]] for x in temp]

        if num_call > 100:
            if i % int(num_call/100) == 0:
                print(round(i/int(num_call/100), 2))

    result = sorted(data, key=lambda x: x[0])

    return result

## data = get_data('BTCUSDT', '1m', 3)

data = pd.read_csv('eth_price.csv', index_col=0)

def get_vol_by_window(data, window_size, vol_adjust):
    result = []
    prices = np.array(data.price)
    for i in range(len(prices)-window_size):
        temp = prices[i:i+window_size]
        pct_change = np.diff(temp) / temp[:-1]
        vol = np.std(pct_change) * np.sqrt(vol_adjust)
        result.append(vol)
        if i % int((len(prices)-window_size)/100) == 0:
            print(round(i / int((len(prices)-window_size)/ 100), 2))
    return result

# times = np.array(data.time)
# utc = [datetime.utcfromtimestamp(x/1000) for x in times]

def plotting(vols, utc):
    t = utc[-len(vols):]
    plt.plot(t, vols)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

def classification(target, window_size):
    result = []
    for i in range(len(target)-window_size-1):
        mean = np.mean(target[i:i+window_size])
        std = np.std(target[i:i+window_size])

        if target[i+window_size] > mean + 3*std:
            result.append(i+window_size)
        elif target[i+window_size] < mean - 3*std:
            result.append(i + window_size)

        if i % int((len(target)-window_size)/100) == 0:
            print(round(i / int((len(target)-window_size)/ 100), 2))

    return result

def generate_price(drift, vol, num_step):
    prices = [1]
    for _ in range(num_step):
        pct_change = drift + vol * np.random.normal()
        price = prices[-1] * np.exp(pct_change)
        prices.append(price)
    return prices

def generate_price_jd(drift, vol, jump_prob, jump_mean, jump_vol, num_step):
    prices = [1]
    for _ in range(num_step):
        if random.random() < jump_prob:
            jump = jump_mean + jump_vol * np.random.normal()
        else:
            jump = 0
        pct_change = drift + vol * np.random.normal() + jump
        price = prices[-1] * np.exp(pct_change)
        prices.append(price)
    return prices

def touch_real(prices, num_step, barrier):
    count = 0
    for i in range(len(prices)-num_step):
        temp = prices[i:i+num_step]
        if max(temp) > temp[0] * (1 + barrier) or min(temp) < temp[0] * (1 - barrier):
            count += 1
        if i % int((len(prices)-num_step)/100) == 0:
            print(round(i / int((len(prices)-num_step)/ 100), 2))
    return count

def touch_simul(drift, vol, num_step, barrier, num_simul):
    count = 0
    for i in range(num_simul):
        price = 1
        for _ in range(num_step):
            pct_change = drift + vol * np.random.normal()
            price *= np.exp(pct_change)
            if price > 1 + barrier or price < 1 - barrier:
                count += 1
                break
    return count

def touch_simul_jd(drift, vol, jump_prob, jump_mean, jump_vol, num_step, barrier, num_simul):
    count = 0
    for i in range(num_simul):
        price = 1
        for _ in range(num_step):
            if random.random() < jump_prob:
                jump = jump_mean + jump_vol * np.random.normal()
            else:
                jump = 0
            pct_change = drift + vol * np.random.normal() + jump
            price *= np.exp(pct_change)
            if price > 1 + barrier or price < 1 - barrier:
                count += 1
                break
    return count





