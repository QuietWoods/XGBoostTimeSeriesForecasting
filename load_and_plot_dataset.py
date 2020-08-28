# -*- coding: utf-8 -*-
# @Time    : 2020/8/26 22:51
# @Author  : QuietWoods
# @FileName: load_and_plot_dataset.py
# @Software: PyCharm

# load and plot the time series dataset
from pandas import read_csv
from matplotlib import pyplot
# load dataset
series = read_csv('data/daily-total-female-births.csv', header=0, index_col=0)
values = series.values
# plot dataset
pyplot.plot(values)
pyplot.show()