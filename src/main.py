import datetime
from yahoo_fin.stock_info import get_data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import cv2

def pre_process_data():
    pass

"""
Getting Stock Data From Yahoo Finance
Count by year
"""
def get_data_ticker(ticker, period=1):
    ago = datetime.datetime.now() - datetime.timedelta(days=period*365)
    return get_data(ticker, start_date=ago.strftime('%d/%m/%Y'), index_as_date=True, interval="1d")


fig = plt.figure()
ticker = 'ARKK'
dataframe = get_data_ticker(ticker, 1)
plt.plot(dataframe.index, dataframe['close'])
plt.show()
fig.canvas.draw()
image = np.array(fig.canvas.get_renderer()._renderer)
print(image)
cv2.imshow("image", image)