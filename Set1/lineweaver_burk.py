import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from michaelis_menten import michaelis_menten_equation

with open('Data/measuredData.csv', 'r') as f:
    reader = csv.reader(f)

    data_list = []
    for row in reader:
        data_list.append([float(row[0]), float(row[1])])
    # data_list = list(reader)
    data_list = np.transpose(data_list)
    # print(np.transpose(data_list))

def make_fit():
    popt, pcov = curve_fit(michaelis_menten_equation, data_list[0], data_list[1])

    x_values = np.linspace(min(data_list[0]), max(data_list[0]), 100)
    plt.plot(x_values, michaelis_menten_equation(x_values, popt[0], popt[1]), label="Fit")
    plt.plot(data_list[0], data_list[1], label="Measured data")
    plt.xlabel("[S]")
    plt.ylabel("[V]")
    plt.legend()
    plt.show()

    return popt

def linear_fit_func(x, a, b):
    return a + b*x

def LB_params():
    popt, pcov = curve_fit(linear_fit_func, 1/data_list[0][1:], 1/data_list[1][1:])

    return [1 / popt[0], popt[1] / popt[0]]

def LB_plot():
    plt.plot(1/data_list[0][1:], 1/data_list[1][1:])
    plt.xlabel("1/[S]")
    plt.ylabel("1/[V]")
    plt.show()

# print(make_fit(), LB_params())