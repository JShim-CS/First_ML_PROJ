import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Normalization, Input
from tensorflow.keras.optimizers import Adam
from sklearn import preprocessing
import matplotlib.pyplot as plt

import os
import numpy as np
import pandas as pd
import random



def make_models(Open_Price_normalizer):

    model1 = Sequential([Open_Price_normalizer,
    Dense(254,activation = "relu"),
    Dense(254,activation = "relu"),
    Dense(units = 1)])

    model2 = Sequential([Open_Price_normalizer,
    Dense(254,activation = "relu"),
    Dense(254,activation = "relu"),
    Dense(units = 1)])

    return model1, model2

def compile_models(model1, model2):
    model1.compile(optimizer = Adam(0.001), loss = "mean_absolute_error")
    model2.compile(optimizer = Adam(0.001), loss = "mean_absolute_error")

def history_summaries(model1, model2):
    history1 = model1.fit(
    x_train,
    y_train1,
    epochs = 100,
    verbose = 0,
    validation_split = 0.2
    )
    model1.summary()

    hist1 = pd.DataFrame(history1.history)
    hist1['epoch'] = history1.epoch
    hist1.tail()
    print(hist1)

    history2 = model2.fit(
    x_train,
    y_train2,
    epochs = 100,
    verbose = 0,
    validation_split = 0.2
    )
    model2.summary()

    hist2 = pd.DataFrame(history2.history)
    hist2['epoch'] = history2.epoch
    hist2.tail()
    print(hist2)

def evaluate_models(model1,model2):
    m1 = model1.evaluate(x_test, y_test1, verbose = 0)
    m2 = model2.evaluate(x_test, y_test2, verbose = 0)

def predict_models(model1,model2,value):
    print(model1.predict(np.asarray([value])))
    print(model2.predict(np.asarray([value])))



def plot_low1(x,y):
    plt.scatter(x_train,y_train1, label = "High")
    plt.plot(x,y,color="k",label="Predictions")
    plt.xlabel("Open")
    plt.ylabel("High")
    plt.legend()
    plt.show()

def plot_low2(x,y):
    plt.scatter(x_train,y_train2, label = "Low")
    plt.plot(x,y,color="k",label="Predictions")
    plt.xlabel("Open")
    plt.ylabel("Low")
    plt.legend()
    plt.show()




def plot_graphs(model1, model2):
    x1 = tf.linspace(0.0,np.max(x_train), np.max(x_train) + 1)
    y1 = model1.predict(x1)
    plot_low1(x1,y1)

    x2 = tf.linspace(0.0,np.max(x_train), np.max(x_train) + 1)
    y2 = model2.predict(x1)
    plot_low2(x2,y2)

if __name__ == "__main__":
    np.random.seed(314)
    tf.random.set_seed(314)
    random.seed(314)

    col_names = ["Date", "Open","High","Low","Close","Volume","OpenInt"]

    raw_train_data = pd.read_csv("amzn_us_train.csv",
                                 names = col_names, na_values = "?",
                                 comment = '\t', sep=',', skiprows = [0])

    raw_test_data = pd.read_csv("amzn_us_test.csv",
                                 names = col_names, na_values = "?",
                                 comment = '\t', sep=',', skiprows = [0])


    x_train = raw_train_data["Open"]
    y_train1 = raw_train_data["High"]
    y_train2 = raw_train_data["Low"]

    x_test = raw_test_data["Open"]
    y_test1 = raw_test_data["High"]
    y_test2 = raw_test_data["Low"]

    Open_Price = np.array(x_train)

    Open_Price_normalizer = Normalization(input_shape = [1,], axis = None)
    Open_Price_normalizer.adapt(Open_Price)

    model1, model2 = make_models(Open_Price_normalizer)
    compile_models(model1,model2)
    history_summaries(model1, model2)
    #evaluate_models(model1, model2)
    predict_models(model1, model2, 1.97)
    plot_graphs(model1,model2)
