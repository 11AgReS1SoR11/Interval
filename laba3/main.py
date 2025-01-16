from collections import defaultdict
import numpy as np
np.float_ = np.float64
import intvalpy as ip
import matplotlib.pyplot as plt
from functools import cmp_to_key

from data_preparation import GetData

ip.precision.extendedPrecisionQ = False


def union_intervals(x, y):
    return ip.Interval(min(x.a, y.a), max(x.b, y.b))


def mode(X):
    print("Calculate mode")
    if X is None:
        return None

    Y = []
    for el in X:
        Y.append(el.a)
        Y.append(el.b)

    Y.sort()

    Z = [ip.Interval(Y[i], Y[i + 1]) for i in range(len(Y) - 1)]

    mu = [sum(1 for x_i in X if z_i in x_i) for z_i in Z]

    max_mu = max(mu)
    K = [index for index, element in enumerate(mu) if element == max_mu]

    m = [Z[k] for k in K]
    mode_ = []

    current_interval = m[0]

    for next_interval in m[1:]:
        print(current_interval, next_interval)
        res_inter = ip.intersection(current_interval, next_interval)
        if not (np.isnan(res_inter.a) and np.isnan(res_inter.b)):
            current_interval = union_intervals(current_interval, next_interval)
        else:
            mode_.append(current_interval)
            current_interval = next_interval

    mode_.append(current_interval)

    return np.array(mode_)


def mode_1(X):
    if X is None:
        return None

    Y = np.array([el for sub in X for el in (sub.a, sub.b)])
    Y.sort()

    Z = [ip.Interval(Y[i], Y[i + 1]) for i in range(len(Y) - 1)]

    mu = defaultdict(int)
    for z_i in Z:
        for x_i in X:
            if z_i in x_i:
                mu[(z_i.a, z_i.b)] += 1

    max_mu = max(mu.values())
    m = [ip.Interval(a, b) for (a, b), count in mu.items() if count == max_mu]

    mode_ = []
    current_interval = m[0]

    for next_interval in m[1:]:
        res_inter = ip.intersection(current_interval, next_interval)
        if not (np.isnan(res_inter.a) and np.isnan(res_inter.b)):
            current_interval = union_intervals(current_interval, next_interval)
        else:
            mode_.append(current_interval)
            current_interval = next_interval

    mode_.append(current_interval)

    return np.array(mode_)


def med_K(X_data):
    c_inf = [ip.inf(el) for el in X_data]
    c_sup = [ip.sup(el) for el in X_data]
    return ip.Interval(np.median(c_inf), np.median(c_sup))


def med_P(X_data):
    x = sorted(X_data, key=cmp_to_key(lambda x, y: (x.a + x.b) / 2 - (y.a + y.b) / 2))
    index_med = len(x) // 2
    if len(x) % 2 == 0:
        return (x[index_med - 1] + x[index_med]) / 2
    return x[index_med]


def coefficient_Jakkard(X_data, Y_data=None):
    if Y_data is None:
        x_inf = [ip.inf(x) for x in X_data]
        x_sup = [ip.sup(x) for x in X_data]
        return (min(x_sup) - max(x_inf)) / (max(x_sup) - min(x_inf))


def argmaxF(f, a, b, eps):
    lmbd = a + (3 - 5 ** 0.5) * (b - a) / 2
    mu = b - (3 - 5 ** 0.5) * (b - a) / 2
    f_lambda = f(lmbd)
    f_mu = f(mu)

    while 1:
        if f_lambda <= f_mu:
            a = lmbd
            if eps > b - a:
                break
            lmbd = mu
            f_lambda = f_mu
            mu = b - (3 - 5 ** 0.5) * (b - a) / 2
            f_mu = f(mu)
        else:
            b = mu
            if eps > b - a:
                break
            mu = lmbd
            f_mu = f_lambda
            lmbd = a + (3 - 5 ** 0.5) * (b - a) / 2
            f_lambda = f(lmbd)

        # print(a)
        # print(b)

    return (a + b) / 2


def func_a(a):
    new_X = X + a
    XY = np.concatenate((new_X, Y))
    return coefficient_Jakkard(XY)


def func_t(t):
    new_X = X*t
    XY = np.concatenate((new_X, Y))
    return coefficient_Jakkard(XY)


def func_mode_a(a):
    # new_X = X + a
    # mode_X = mode(new_X)
    XY = np.concatenate((mode_X + a, mode_Y))
    return coefficient_Jakkard(XY)


def func_mode_t(t):
    # new_X = X*t
    # mode_X = mode(new_X)
    XY = np.concatenate((mode_X*t, mode_Y))
    return coefficient_Jakkard(XY)


def func_med_p_a(a):
    XY = np.array([med_P_X + a, med_P_Y])
    return coefficient_Jakkard(XY)


def func_med_p_t(t):
    XY = np.array([med_P_X*t, med_P_Y])
    return coefficient_Jakkard(XY)


def func_med_k_a(a):
    XY = np.array([med_K_X + a, med_K_Y])
    return coefficient_Jakkard(XY)


def func_med_k_t(t):
    XY = np.array([med_K_X*t, med_K_Y])
    return coefficient_Jakkard(XY)


def draw_func(f, a, b, parametr: str, func=""):
    if parametr == "a":
        X_linsp = np.linspace(a, b, 1000)
    else:
        X_linsp = np.linspace(a, b, 1000)
    y = [f(x) for x in X_linsp]
    y_max = max(y)
    y_min = min(y)
    ind_max = y.index(y_max)
    ind_min = y.index(y_min)
    x_max = X_linsp[ind_max]
    print(x_max, y_max)

    # import scipy
    # from scipy.interpolate import interp1d
    # f1 = interp1d(np.array(y[0:ind_max + 1]), np.array(X_linsp[0:ind_max + 1]), kind='linear')
    # f2 = interp1d(np.array(y[ind_max:]), np.array(X_linsp[ind_max:]), kind='linear')
    #
    # if y_max < 0:
    #     int_line = (3 * y[ind_max] + y[ind_min]) / 4
    #     ext_line = (y[ind_max] + 3 * y[ind_min]) / 4
    # else:
    #     ext_line = 0.5
    #     int_line = 0.8
    #
    # print(f"internal: {int_line}, x=[{f1(int_line)},{f2(int_line)}]")
    # print(f"external: {ext_line}, x=[{f1(ext_line)},{f2(ext_line)}]")

    # plt.figure(figsize=(12, 9))
    plt.figure(figsize=(10, 6))
    plt.plot(X_linsp, y, color='red')
    plt.xlabel(f"{parametr}, {parametr}_max={round(x_max, 5)}")
    plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    plt.axvline(x=x_max, linestyle='--', color='black')

    # plt.hlines(int_line, f1(int_line), f2(int_line), linestyles='dashed', label='internal', colors='green')
    # plt.hlines(ext_line, f1(ext_line), f2(ext_line), linestyles='dashed', label='external', colors='red')
    # plt.legend()
    plt.legend()
    plt.grid(True)
    plt.title("Jaccard Index")
    plt.savefig(f"Jaccard-{parametr}-{func}")
    plt.show()

def draw_func_all(i, f, a, b, parametr: str, func=""):
    colors = ["#EF476F", "#F78C6B", "#FFD166", "#83D483", "#06D6A0", "#0CB0A9", "#118AB2", "#073B4C"]
    X_linsp = np.linspace(a, b, 100)
    y = np.array([f(x) for x in X_linsp])
    plt.plot(X_linsp, y, color=colors[i], label=f"Ji({parametr}, {func}(X), {func}(Y))", alpha=0.7)

    plt.xlabel(f"{parametr}")
    plt.ylabel(f"Ji({parametr}, {func}(X), {func}(Y))")
    plt.title("Jaccard Index")
    plt.show()
    plt.savefig(f"Jaccadrd-{parametr}-{func}")


if __name__ == "__main__":
    X, Y = GetData()

    # Функционал = Ji(const, X, Y)
    draw_func(func_a, 0, 0.8, "a")
    draw_func(func_t, -1.6, -0.7, "t")

    # a_f = argmaxF(func_a, 0, 1, 1e-3)
    # print(a_f, func_a(a_f))
    # t_f = argmaxF(func_t, -4, 0, 1e-3)
    # print(t_f, func_t(t_f))

    # Функционал = Ji(const,mode(X), mode(Y))

    mode_X = mode(X)
    mode_Y = mode(Y)
    draw_func(func_mode_a, 0.3469, 0.34695, "a", "mode")
    draw_func(func_mode_t, -1.0398, -1.0394, "t", "mode")

    # a_f_mode = argmaxF(func_mode_a, 0, 1, 1e-3)
    # print(a_f_mode, func_mode_a(a_f_mode))
    # t_f_mode = argmaxF(func_mode_t, -4, 0, 1e-3)
    # print(t_f_mode, func_mode_t(t_f_mode))

    # Функционал = Ji(const,med_K(X), med_K(Y))
    med_K_X = med_K(X)
    med_K_Y = med_K(Y)
    draw_func(func_med_k_a, 0.2, 0.45, "a", "med_K")
    draw_func(func_med_k_t, -1, -1.1, "t", "med_K")

    # a_f_med_k = argmaxF(func_med_k_a, 0, 1, 1e-3)
    # print(a_f_med_k, func_med_k_a(a_f_med_k))
    # t_f_med_k = argmaxF(func_med_k_t, -4, 0, 1e-3)
    # print(t_f_med_k, func_med_k_t(t_f_med_k))

    # Функционал = Ji(const,med_р(X), med_р(Y))
    med_P_X = med_P(X)
    med_P_Y = med_P(Y)
    draw_func(func_med_p_a, 0.2, 0.45, "a", "med_p")
    draw_func(func_med_p_t, -1, -1.1, "t", "med_p")

    # a_f_med_p = argmaxF(func_med_p_a, 0, 1, 1e-3)
    # print(a_f_med_p, func_med_p_a(a_f_med_p))
    # t_f_med_p = argmaxF(func_med_p_t, -4, 0, 1e-3)
    # print(t_f_med_p, func_med_p_t(t_f_med_p))

    # funcs = [func_a, func_t, func_a_med_k, func_t_med_k, func_a_med_p, func_t_med_p]
    # funcs_str = ["", "", "med_k", "med_k", "med_p", "med_p"]
    # bounds = [[0, 1], [-4, 0], [0, 1], [-4, 0], [0, 1], [-4, 0]]
    # params = ["a", "t", "a", "t", "a", "t"]
    #
    # for i in range(1, len(funcs)+1, 2):
    #     draw_func_all(i, funcs[i], bounds[i][0], bounds[i][1], params[i], funcs_str[i])
    # plt.xlabel(f"const")
    # plt.ylabel(f"Ji(const, func(X), func(Y))")
    # plt.title("Jaccard Index")
    # plt.legend()
    # plt.savefig(f"Jaccadrd-all-in-one-T")
    # plt.show()
    #
    # for i in range(0, len(funcs), 2):
    #     draw_func_all(i, funcs[i], bounds[i][0], bounds[i][1], params[i], funcs_str[i])
    # plt.xlabel(f"const")
    # plt.ylabel(f"Ji(const, func(X), func(Y))")
    # plt.title("Jaccard Index")
    # plt.legend()
    # plt.savefig(f"Jaccadrd-all-in-one-A")
    # plt.show()
