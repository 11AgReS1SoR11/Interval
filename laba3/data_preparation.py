import struct
import numpy as np
import intvalpy as ip
import matplotlib.pyplot as plt


def read_bin_file_with_numpy(file_path):
    with open(file_path, 'rb') as f:
        header_data = f.read(256)
        side, mode_, frame_count = struct.unpack('<BBH', header_data[:4])

        frames = []
        point_dtype = np.dtype('<8H')

        for _ in range(frame_count):
            frame_header_data = f.read(16)
            stop_point, timestamp = struct.unpack('<HL', frame_header_data[:6])
            frame_data = np.frombuffer(f.read(1024 * 16), dtype=point_dtype)
            frames.append(frame_data)
        print("Complete load data")
        return np.array(frames)


def scalar_to_interval(x, rad):
    return ip.Interval(x - rad, x + rad)


scalar_to_interval_vec = np.vectorize(scalar_to_interval)


def boxplot_T(data_list):
    LQ = np.quantile(data_list, 0.25)
    UQ = np.quantile(data_list, 0.75)
    IQR = UQ - LQ
    s = sorted(data_list)
    x_L = max(s[0], LQ - 3 / 2 * IQR)
    x_U = min(s[-1], UQ + 3 / 2 * IQR)

    return np.array([x_L, x_U])


def checking_for_anomaly(data_list, ranges):
    idx = []
    A = []
    for i in range(len(data_list)):
        row = [f"x({i + 1})", f"{data_list[i]}"]
        if data_list[i] not in ip.Interval(ranges):
            row.append("аномальна")
            idx.append(i)
            A.append(row)
    return idx


def remove_outliers(data_list):
    ranges = boxplot_T(data_list)
    idx = checking_for_anomaly(data_list, ranges)
    data_list = np.delete(data_list, idx)
    return data_list


def get_avg(data):
    avg = [[0]*8]*1024
    for i in range(len(data)): # 100
        avg = np.add(avg, data[i])
    return np.divide(avg, len(data))


def GetData():
    x_data = read_bin_file_with_numpy('-0.205_lvl_side_a_fast_data.bin')
    y_data = read_bin_file_with_numpy('0.225_lvl_side_a_fast_data.bin')

    x_data = get_avg(x_data)
    y_data = get_avg(y_data)

    x_voltage = x_data / 16384.0 - 0.5
    y_voltage = y_data / 16384.0 - 0.5

    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.hist(x_voltage.flatten(), bins=100)
    plt.title("X до обработки")
    plt.subplot(1, 2, 2)
    plt.hist(y_voltage.flatten(), bins=100)
    plt.title("Y до обработки")
    plt.savefig(f"XYbefore")
    plt.show()

    x_after = remove_outliers(x_voltage.flatten())
    y_after = remove_outliers(y_voltage.flatten())

    plt.figure(figsize=(12, 9))
    plt.subplot(1, 2, 1)
    plt.hist(x_after, bins=100)
    plt.title("X после отбрасывания выбросов")
    plt.subplot(1, 2, 2)
    plt.hist(y_after, bins=100)
    plt.title("Y после отбрасывания выбросов")
    plt.savefig(f"XYafter")
    plt.show()

    rad = 2 ** (-14)

    X = scalar_to_interval_vec(x_after, rad).flatten()
    Y = scalar_to_interval_vec(y_after, rad).flatten()

    print("Convert X and Y")
    return X, Y

