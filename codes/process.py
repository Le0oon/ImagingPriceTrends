# from Wind_Process import *
# 用20天的label生成60天的图像（中证500）
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
import multiprocessing as mp
from PIL import Image
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import subprocess

data = pd.read_csv("data_500.csv")


class OHLC():
    """
    plot the OHLC charts
    """
    import matplotlib.pyplot as plt

    def __init__(self,
                 data,
                 ma_cols=['5_day_MA', '20_day_MA', '60_day_MA'],
                 ):
        self.data = data.copy()
        self.feat_cols = ['open', 'high', 'low', 'close'] + ma_cols
        self.ma_cols = ma_cols

    def standardize_data(self):
        df = self.data.copy()
        # df = df.assign(date = pd.to_datetime(df.date,format = "%Y%m%d"))
        df = df[['code', 'date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'ret_f1', 'ret_cf2', 'ret_cf3',
                 'ret_cf4', 'ret_cf5', 'ret_cf10', 'ret_cf20', 'ret_cf60', '5_day_MA', '20_day_MA', '60_day_MA']]
        df.reset_index(inplace=True)
        # we normalize the open price to be one at the initial date
        scale = df.loc[0, 'open']
        for col in ['open', 'high', 'close', 'low', '5_day_MA', '20_day_MA', '60_day_MA']:
            df[col] = df[col] / scale
        # we normalize the volume to be one at the initial date
        scale = df.loc[0, 'volume']
        df['volume'] = df['volume'] / scale
        return df

    def plot(self):
        df = self.data.reset_index(drop=True).copy()
        high = df[['high'] + self.ma_cols].values.max()
        low = df[['low'] + self.ma_cols].values.min()
        module = high - low
        if module is None or module == 0 or np.isnan(module):
            return np.zeros((96, 180))
        for col in self.feat_cols:
            df[col] = (df[col] - low) / module * 76
            df[col] = df[col].round(0).astype(int) + 19

        df['volume'] = df['volume'] / df['volume'].max() * 19
        df['volume'] = df['volume'].round(0).astype(int)

        plot = np.zeros((96, 180))
        for idx, row in df.iterrows():
            plot[row['open'], idx*3] = 1
            plot[row['low']:row['high']+1, idx*3+1] = 1
            plot[row['close']:row['close']+1, idx*3+2] = 1
            plot[:row['volume']+1, idx*3+1] = 1

            for col in self.ma_cols:
                pre_ma = df.loc[idx-1, col] if idx >= 1 else df.loc[idx, col]
                next_ma = df.loc[idx+1, col] if idx <= len(df)-2 else df.loc[idx, col]

                plot[(row[col] + pre_ma)//2, idx*3] = 1
                plot[row[col], idx*3+1] = 1
                plot[(row[col] + next_ma)//2, idx*3+2] = 1

        # plot *= 255
        plot = plot[::-1, :]  # 反转y 轴

        return plot  # np.array with shape (96,180)


os.makedirs("../data_base", exist_ok=True)
os.chdir("../data_base")


class Sample:
    def __init__(self, data, codes, dates, calendars, seq_length=60, train=True):
        self.data = data.copy()
        self.codes = codes
        self.dates = dates
        self.train = train
        self.seq_length = seq_length
        self.calendars = calendars

    def sample_one_img(self, code, date):  # 这里的date是最后一天的date
        idx = self.calendars.index(date)
        start_idx = idx - self.seq_length + 1
        start_date = self.calendars[start_idx]
        df = self.data.loc[self.data.code == code]
        df = df.loc[(df.date >= start_date) & (df.date <= date)]

        if date not in df.date.values or len(df) < self.seq_length or df.volume.min() == 0:
            return None

            # 生成图片
        ohlc = OHLC(df,['60_day_MA'])
        fig = ohlc.plot()
        fig = (fig * 255).astype(np.uint8)
        rgb_image = np.zeros((96, 180,3), dtype=np.uint8)
        for channel in range(3):
            rgb_image[:, :, channel] = fig
        image = Image.fromarray(rgb_image, 'RGB')

        fig_id = code + "-" + str(date)
        ret = df.loc[(df['code'] == code) & (df['date'] == date), 'ret_cf20'].values[0]
        # print(ret)
        if self.train == True:
            if ret > 0:
                os.makedirs("train/1", exist_ok=True)
                fig_name = "train/1/" + fig_id + ".png"
            if ret <= 0:
                os.makedirs("train/0", exist_ok=True)
                fig_name = "train/0/" + fig_id + ".png"
        else:
            if ret > 0:
                os.makedirs("test/1", exist_ok=True)
                fig_name = "test/1/" + fig_id + ".png"
            if ret <= 0:
                os.makedirs("test/0", exist_ok=True)
                fig_name = "test/0/" + fig_id + ".png"
        image.save(fig_name)

        # plt.clf()

        df = df[['date', 'ret_f1', 'ret_cf2', 'ret_cf3', 'ret_cf4', 'ret_cf5',
                 'ret_cf10', 'ret_cf20', 'ret_cf60']]
        df['id'] = fig_id

        df = df[df.date == date]
        return df

    def sample_batch_imgs(self):

        i = 0
        for code in self.codes:
            if self.train:
                sample_size = int(len(self.dates) * 0.2)
                dates = random.sample(self.dates, sample_size)
            else:
                dates = self.dates
            for date in dates:
                print("code:", code, "date:", date)
                try:
                    if i == 0:
                        result = self.sample_one_img(code, date)
                        if result is not None:
                            i += 1
                            df = result.copy()
                    else:
                        result = self.sample_one_img(code, date)
                        if result is not None:
                            i += 1
                            df = df.append(result)
                except:
                    pass

        return df

    def random_sample(self, code, date):
        if self.train:
            # only random draw 20% of sample, I use this to support parallel process
            # if np.random.rand() <= 0.2:
            return self.sample_one_img(code, date)
            # print(code, date)
        else:
            return self.sample_one_img(code, date)


# generate train samples
data = data[data['class'] == 0]
calendars = list(data.date.unique())
codes = list(data.code.unique())
# dates = list(data[data.date <= 20130101].date.unique()) ## 确定train sample 的日期
codes.sort()

# dates.sort()
calendars.sort()
#
# sample = Sample(data,codes,dates,calendars,64,True)
# sample = sample.sample_batch_imgs()

dates = data.date.unique()
dates = [20130131, 20130228, 20130329, 20130426, 20130531, 20130628, 20130731, 20130830, 20130930, 20131031, 20131129, 20131231, 20140130, 20140228, 20140331, 20140430, 20140530, 20140630, 20140731, 20140829, 20140930, 20141031, 20141128, 20141230]
dates.sort()

sample = Sample(data, codes, dates, calendars, 60, False)  # 如果是train集，则随机抽取1/5的数据生成图像
# sample.sample_batch_imgs()

# from joblib import Parallel,delayed
# results = Parallel(n_jobs = 10,prefer = 'threads')(delayed(sample.sample_batch_imgs))

results = Parallel(n_jobs=16, prefer='processes', verbose=10)(
    delayed(sample.random_sample)(code, date) for code in codes for date in dates)
