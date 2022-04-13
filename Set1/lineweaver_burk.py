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

    # switch columns and rows
    data_list = np.transpose(data_list)

def make_fit():
    # obtain fitting parameters
    popt, pcov = curve_fit(michaelis_menten_equation, data_list[0], data_list[1])

    # plot the fit together with the measurements
    x_values = np.linspace(min(data_list[0]), max(data_list[0]), 100)
    plt.plot(x_values, michaelis_menten_equation(x_values, popt[0], popt[1]), 'r-', label="Fit")
    plt.scatter(data_list[0], data_list[1], label="Measured data")
    plt.xlabel("[S]")
    plt.ylabel("v")
    plt.legend()
    plt.show()

    return popt

def linear_fit_func(x, a, b):
    # function for a linear relationship
    return a + b*x

def LB_params():
    # obtain MM parameters using the intersections of the linear fit in the LB plot
    popt, pcov = curve_fit(linear_fit_func, 1/data_list[0][1:], 1/data_list[1][1:])

    return [1 / popt[0], popt[1] / popt[0]]

def LB_plot():
    # obtain fitting parameters
    popt, pcov = curve_fit(linear_fit_func, 1/data_list[0][1:], 1/data_list[1][1:])

    # plot a and b
    print(popt)

    # plot the fit together with the measurements
    x_values = np.linspace(min(1/data_list[0][1:]), max(1/data_list[0][1:]), 100)
    plt.plot(x_values, linear_fit_func(x_values, popt[0], popt[1]), 'r-', label="Fit")
    plt.scatter(1/data_list[0][1:], 1/data_list[1][1:], label="Measured data")
    plt.xlabel("1/[S]")
    plt.ylabel("1/v")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    LB_plot()