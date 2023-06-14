import adi
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.fft as fft
import scipy.signal as signal
from pandas import read_csv, DataFrame
import matplotlib as mpl
import functools
from timeit import default_timer as timer

sdr = adi.ad9361('ip:172.16.1.246')
# %%
sdr.tx_destroy_buffer()
sdr.rx_lo = int(100e6)
sdr.tx_lo = int(100e6)
sdr.sample_rate = int(61.44e6)
sdr.rx_rf_bandwidth = int(1.6 * sdr.sample_rate / 2)
sdr.tx_rf_bandwidth = int(1.6 * sdr.sample_rate / 2)
sdr.gain_control_mode = "slow_attack"
sdr.tx_enabled_channels = [1]
sdr.rx_enabled_channels = [1]
sdr.tx_buffer_size = 8192
sdr.rx_buffer_size = 8192
# Подключение: TX2 - RX1
# sdr.tx_destroy_buffer()
sdr.tx_cyclic_buffer = True
sdr.loopback = 0

print("Values:")
blacklist = ["gain_table_config", "multichip_sync"]
for dev in sdr.ctx.devices:
    print(dev.name, "{")
    if dev.name == "ams" or dev.name == "ad7291" or dev.name == None:
        continue
    for attr in dev.attrs:
        attr = dev.attrs[attr]
        print("\t",attr.name,end=" ")
        if attr.name in blacklist:
            print()
        else:
            print(attr.value)
    print("Channel attrs:")
    for channel in dev.channels:
        print("\t",channel.name, "{")
        for attr in channel.attrs:
            attr = channel.attrs[attr]
            print("\t\t",attr.name,end=" ")
            if attr.name in blacklist:
                print()
            else:
                print(attr.value)
        print("\t}")
    print("}")


