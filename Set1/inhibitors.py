import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from michaelis_menten import michaelis_menten_equation
from lineweaver_burk import linear_fit_func

def competitive_inhibitor(S, I, Vmax, Km, Ki):
    return Vmax * S / ((1 + I / Ki) * Km + S)

def uncompetitive_inhibitor(S, I, Vmax, Km, Ki):
    return Vmax * S / (Km + (1 + I / Ki) * S)

def noncompetitive_inhibitor(S, I, Vmax, Km, Ki):
    return Vmax * S / ((1 + I / Ki) * (Km + S))

def inverse_com(inverse_S, I, Vmax, Km, Ki):
    return 1/Vmax + ((1 + I / Ki) * Km) / Vmax * inverse_S

def inverse_uncom(inverse_S, I, Vmax, Km, Ki):
    return (1 + I/Ki) / Vmax + Km / Vmax * inverse_S

def inverse_noncom(inverse_S, I, Vmax, Km, Ki):
    return (1 + I/Ki) / Vmax + ((1 + I / Ki) * Km) / Vmax * inverse_S

def e(I):
    with open(f'Data/DataI{I}.csv', 'r') as f:
        reader = csv.reader(f)

        data_list = []
        for row in reader:
            data_list.append([float(row[0]), float(row[1])])
        
        data_list = np.transpose(data_list)

    plt.plot(1/data_list[0][1:], 1/data_list[1][1:], label="Data")
    i = 0
    for inverse_inhibitor in (inverse_com, inverse_uncom, inverse_noncom):
        popt, pcov = curve_fit(lambda S, Vmax, Km, Ki: inverse_inhibitor(S, I, Vmax, Km, Ki), 1/data_list[0][1:], 1/data_list[1][1:], p0=(1,1,1), bounds=((0, 0, 0), (10**8, 10**8, 10**8)))
        print(f"Vmax = {popt[0]}, Km = {popt[1]}, Ki= {popt[2]}")
        print(f"perr = {np.sqrt(np.diag(pcov))}")
        x_values = np.linspace(min(1/data_list[0][1:]), max(1/data_list[0][1:]), 100)
        plt.plot(x_values, inverse_inhibitor(x_values, I, popt[0], popt[1], popt[2]), ("-", "--", ":")[i%3], label=f"Fit {inverse_inhibitor.__name__}")
        i += 1
    print("")
    plt.xlabel("1/[S]")
    plt.ylabel("1/[V]")
    plt.title(f"I = {I}")
    plt.legend()
    plt.show()

    # popt, pcov = curve_fit(lambda S, Vmax, Km, Ki: competitive_inhibitor(S, 2, Vmax, Km, Ki), data_list[0], data_list[1])
    # print(popt, pcov)
    # x_values = np.linspace(min(data_list[0]), max(data_list[0]), 100)
    # plt.plot(x_values, competitive_inhibitor(x_values, 2, popt[0], popt[1], popt[2]), label="Fit")
    # plt.plot(data_list[0], data_list[1], label="Measured data")
    # plt.xlabel("[S]")
    # plt.ylabel("[V]")
    # plt.legend()
    # plt.show()

def d():
    I_values = np.linspace(1, 20, 5)
    Ki_values = np.linspace(0.2, 4, 5)
    S_values = np.linspace(0, 20, 100)

    inhibitor = competitive_inhibitor

    plt.plot(S_values, michaelis_menten_equation(S_values, 12, 1), label="Michaelis-Menten")
    for I in I_values:
        plt.plot(S_values, inhibitor(S_values, I, 12, 1, 1), label=f"Competitive, I = {I}")
    plt.legend()
    plt.xlabel("[S]")
    plt.ylabel("v")
    plt.show()

    plt.plot(S_values, michaelis_menten_equation(S_values, 12, 1), label="Michaelis-Menten")
    for Ki in Ki_values:
        plt.plot(S_values, inhibitor(S_values, 1, 12, 1, Ki), label=f"Competitive, Ki = {Ki}")
    plt.legend()
    plt.xlabel("[S]")
    plt.ylabel("v")
    plt.show()

if __name__ == "__main__":
    e(2)
    e(5)
    e(8)