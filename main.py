import requests
import numpy as np
import os
os.environ['MPLBACKEND'] = 'TkAgg'  # or any other backend you prefer
import matplotlib.pyplot as plt
import pandas as pd
import bisect
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

## data = pd.read_csv('eth_price.csv', index_col=0)

def make_sub_data(data, interval):
    subset_indices = [i for i in range(1, len(data), interval)]
    return data.iloc[subset_indices]

def get_vol_by_window(data, window_size, vol_adjust):
    result = []
    data['price'] = data['price'].astype(float)
    prices = np.array(data.price)
    for i in range(len(prices)-window_size):
        temp = prices[i:i+window_size]
        pct_change = np.diff(temp) / temp[:-1]
        vol = np.std(pct_change) * np.sqrt(vol_adjust)
        result.append(vol)
        if i % int((len(prices)-window_size)/100) == 0:
            print(round(i / int((len(prices)-window_size)/ 100), 2))
    return result

## times = np.array(data.time)
## utc = [datetime.utcfromtimestamp(x/1000) for x in times]

def plotting(vols, utc):
    t = utc[-len(vols):]
    plt.plot(t, vols)
    plt.xlabel('Time')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()

def classification(data, window_size):
    prices = np.array(data.price)
    target = np.diff(prices) / prices[:-1] # pct_change

    result = []
    for i in range(len(target)-window_size-1):
        mean = np.mean(target[i:i+window_size])
        std = np.std(target[i:i+window_size])

        if target[i+window_size] > mean + 3*std:
            result.append(i+window_size)
        elif target[i+window_size] < mean - 3*std:
            result.append(i+window_size)

        if i % int((len(target)-window_size)/100) == 0:
            print(round(i / int((len(target)-window_size)/ 100), 2))

    return result

def jump_statistic(data, jump_index, std_adjust):
    prices = np.array(data.price)
    pct_change = np.diff(prices) / prices[:-1] # pct_change

    print("Total : mean = %f, std: %f" % (np.mean(pct_change), np.std(pct_change) * np.sqrt(std_adjust)))

    jump_pct = pct_change[jump_index]

    print("Jump : mean = %f, std: %f" % (np.mean(jump_pct), np.std(jump_pct) * np.sqrt(std_adjust)))

    no_jump_pct = pct_change[np.logical_not(np.isin(np.arange(len(pct_change)), jump_index))]

    print("No Jump : mean = %f, std: %f" % (np.mean(no_jump_pct), np.std(no_jump_pct) * np.sqrt(std_adjust)))


def num_jump_slide(total_length, sliding_window_size, jump_index):
    result = []
    for i in range(1, int(total_length/sliding_window_size)):
        count = bisect.bisect_right(jump_index, sliding_window_size*(i+1)) - bisect.bisect_left(jump_index, sliding_window_size*i)
        result.append(count)
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

def get_adq_range(maturity, vol, target_KO_prob):
    return 1/1284.67 * (np.log(1/target_KO_prob-1)+3.148) * vol * 2 * np.sqrt(maturity/10/60)

def touch_real(data, num_step, barrier):
    prices = np.array(data.price)
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

def touch_real_at_point(data, bench_size, data_time, game_time):
    count = 0
    prices = np.array(data.price)
    real_pct_change = np.diff(prices) / prices[:-1]  # pct_change
    game_step = int(game_time / data_time)

    for i in range(len(real_pct_change)-bench_size-game_step):
        sub_data = real_pct_change[i:i+bench_size]
        global_vol = np.std(sub_data)
        barrier = get_adq_range(game_time, global_vol * np.sqrt(3600 * 24 * 365 / data_time), 0.5)
        temp = prices[i+bench_size:i+bench_size+game_step]
        if max(temp) > temp[0] * (1 + barrier) or min(temp) < temp[0] * (1 - barrier):
            count += 1
        if i % int((len(real_pct_change)-bench_size-game_step)/100) == 0:
            print(round(i / int((len(real_pct_change)-bench_size-game_step)/ 100), 2))
    return count

def touch_simul_at_point(data, bench_size, data_time, game_time, num_simul):
    result = []
    prices = np.array(data.price)
    real_pct_change = np.diff(prices) / prices[:-1]  # pct_change

    for i in range(int(len(real_pct_change)/bench_size)):
        count = 0
        sub_data = real_pct_change[bench_size*i:bench_size*(i+1)]
        global_vol = np.std(sub_data)
        barrier = get_adq_range(game_time, global_vol*np.sqrt(3600*24*365/data_time), 0.5)
        for _ in range(num_simul):
            price = 1
            for j in range(game_time):
                pct_change = global_vol * np.random.normal() * np.sqrt(1/data_time)
                price *= np.exp(pct_change)
                if price > 1 + barrier or price < 1 - barrier:
                    count += 1
                    break
        if i % int((len(real_pct_change)/bench_size/100)) == 0:
            print(round(i / int(len(real_pct_change)/bench_size/ 100), 2))
        result.append(count)
    return result

def touch_simul_at_point_jd(data, bench_size, data_time, game_time, num_simul):
    result = []
    prices = np.array(data.price)
    real_pct_change = np.diff(prices) / prices[:-1]  # pct_change

    for i in range(int(len(real_pct_change)/bench_size)):
        count = 0
        sub_data = real_pct_change[bench_size*i:bench_size*(i+1)]
        global_vol = np.std(sub_data)

        jump = sub_data[(sub_data > np.mean(sub_data) + 3*np.std(sub_data)) + (sub_data < np.mean(sub_data)- 3*np.std(sub_data))]
        no_jump = sub_data[(sub_data <= np.mean(sub_data) + 3*np.std(sub_data))*(sub_data >= np.mean(sub_data)- 3*np.std(sub_data))]

        jump_prob = len(jump)/bench_size
        jump_vol = np.std(jump)
        no_jump_vol = np.std(no_jump)

        barrier = get_adq_range(game_time, global_vol*np.sqrt(3600*24*365/data_time), 0.5)
        for _ in range(num_simul):
            price = 1
            for j in range(game_time):
                if random.random() < jump_prob:
                    jump = jump_vol * np.random.normal() * np.sqrt(1/data_time)
                else:
                    jump = 0
                pct_change = no_jump_vol * np.random.normal() * np.sqrt(1/data_time) + jump
                price *= np.exp(pct_change)
                if price > 1 + barrier or price < 1 - barrier:
                    count += 1
                    break
        if i % int((len(real_pct_change)/bench_size/100)) == 0:
            print(round(i / int(len(real_pct_change)/bench_size/ 100), 2))
        result.append(count)
    return result

def touch_simul_at_point_jd_param(data, bench_size, data_time):
    result = []
    prices = np.array(data.price)
    real_pct_change = np.diff(prices) / prices[:-1]  # pct_change

    for i in range(int(len(real_pct_change)/bench_size)):
        sub_data = real_pct_change[bench_size*i:bench_size*(i+1)]
        # global_vol = np.std(sub_data)

        jump = sub_data[(sub_data > np.mean(sub_data) + 3*np.std(sub_data)) + (sub_data < np.mean(sub_data)- 3*np.std(sub_data))]
        no_jump = sub_data[(sub_data <= np.mean(sub_data) + 3*np.std(sub_data))*(sub_data >= np.mean(sub_data)- 3*np.std(sub_data))]

        jump_prob = len(jump)/bench_size
        jump_vol = np.std(jump) * np.sqrt(3600*24*365/data_time)
        no_jump_vol = np.std(no_jump) *np.sqrt(3600*24*365/data_time)

        result.append([jump_prob, jump_vol, no_jump_vol])
    return result