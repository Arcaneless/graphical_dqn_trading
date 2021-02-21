import datetime
import math
from yahoo_fin.stock_info import get_data
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# Custom
import helpers
from model import GDQNModel

def pre_process_data():
    pass

"""
Getting Stock Data From Yahoo Finance
Count by days
"""
def get_data_ticker(ticker, end_date=(2021, 1, 1), period=365, interval="1d", logging=False):
    ago = datetime.datetime(*end_date) - datetime.timedelta(days=period)
    start_date = ago.strftime('%m/%d/%Y')
    end_date = f'{end_date[1]}/{end_date[2]}/{end_date[0]}'
    if logging:
        print(f'Fetching stock data ${ticker} from {start_date} to {end_date}')
    return get_data(ticker, start_date=start_date, end_date=end_date, index_as_date=True, interval=interval)

"""
Fetch the ticker image
"""
def fetch_img(ticker, end_date=(2021, 1, 1), period=365, interval="1d", showAxis=False, logging=False):
    fig = plt.figure()
    dataframe = get_data_ticker(ticker, end_date, period, interval, logging)
    plt.plot(dataframe.index, dataframe['close'], 'k')
    if not showAxis:
        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    fig.canvas.draw()
    image = np.array(fig.canvas.buffer_rgba())
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    plt.close(fig)
    return image

"""
Fetch the ground truth
"""
def fetch_stocks_trend_in(ticker, days=5, from_date=(2021, 1, 1)):
    dataframe = get_data_ticker(ticker, from_date, days, "1d")
    diff = dataframe['close'][-1] - dataframe['open'][0]
    print(diff)
    return [int(diff > 0), int(diff < 0)]


"""
Mass fetch stocks data
Fetch by specific day interval
"""
def grab_by_ticker(ticker, start_date=(2020, 5, 1), days=5, period=365, amount=10):
    start_date = datetime.datetime(*start_date)
    images = [
        fetch_img(
            ticker,
            helpers.datetime2tuple(start_date + datetime.timedelta(days=days*x)),
            period=period,
            logging=True
        )
        for x in range(amount)]
    ground_truths = [
        fetch_stocks_trend_in(
            ticker,
            days,
            helpers.datetime2tuple(start_date + datetime.timedelta(days=days*x)),
        )
        for x in range(amount)
    ]

    return images, ground_truths


"""
Build a preview of training set
"""
def show_images_preview(images):
    plt.figure(figsize=(20, 20))

    size = math.ceil(math.sqrt(len(images)))
    for i in range(len(images)):
        plt.subplot(size, size, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        plt.xlabel(f'{i}')
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    plt.show()


ticker = 'AAPL'
years = 1
interval = "1d"
date_of_determination = (2021, 2, 1)
images, truths = grab_by_ticker(ticker, start_date=(2020, 1, 1), period=90, amount=10)
#show_images_preview(images)
images = np.array([x.reshape((x.shape[0], x.shape[1], 1)) for x in images])
truths = np.array(truths)


model = GDQNModel(480, 640)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(images, truths, test_size=0.3, random_state=42)
model.compile_n_run(Xtrain, Xtest, Ytrain, Ytest, 10)