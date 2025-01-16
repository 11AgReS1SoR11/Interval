import numpy as np
np.float_ = np.float64
import intvalpy as ip
from intvalpy import mid, rad
from tolsolvty import tolsolvty
from ir_problem import ir_problem, ir_outer
from ir_plotmodelset import ir_plotmodelset
import matplotlib.pyplot as plt
from read_dir import rawData_instance

ip.precision.extendedPrecisionQ = False


def print_intervals(ys_int, ys_ext, Xs_lvls):
    ys_int_to_plot = [np.average(i) for i in ys_int]
    ys_ext_to_plot = [np.average(i) for i in ys_ext]


    def gen_yi1(ys_int_to_plot):
        return np.abs(ys_int[:, 0] - ys_int_to_plot)

    def gen_yi2(ys_int_to_plot):
        return np.abs(ys_int[:, 1] - ys_int_to_plot)

    def gen_ye1(ys_ext_to_plot):
        return np.abs(ys_ext[:, 0] - ys_ext_to_plot)

    def gen_ye2(ys_ext_to_plot):
        return np.abs(ys_ext[:, 1] - ys_ext_to_plot)

    yerr_int = [
        gen_yi1(ys_int_to_plot),
        gen_yi2(ys_int_to_plot)
    ]
    yerr_ext = [
        gen_ye1(ys_ext_to_plot),
        gen_ye2(ys_ext_to_plot)
    ]

    # ellipse = Ellipse(((Xs_lvls[2]+Xs_lvls[3])/2, (ys_int_to_plot[2]+ys_int_to_plot[3])/2),
    #                   0.1, 1700, color='r', fill=False)
    # ax = plt.gca()
    # ax.add_patch(ellipse)

    plt.errorbar(Xs_lvls, ys_int_to_plot, yerr=yerr_int, marker=".", linestyle='none',
                 ecolor='b', elinewidth=0.8, capsize=4, capthick=1)
    plt.errorbar(Xs_lvls, ys_ext_to_plot, yerr=yerr_ext, linestyle='none',
                 ecolor='orange', elinewidth=0.8, capsize=4, capthick=1)
    # plt.show()


def plot_tol_sys(Xi, Ysint, Ysout, fname, title="Допусковое множество"):
    vert1 = ip.IntLinIncR2(Xi, Ysint, consistency='tol', show=False)
    print(vert1)

    vert = ip.IntLinIncR2(Xi, Ysout, consistency='tol', show=False)
    print(vert)

    plt.figure(figsize=(10, 6))

    for ortant in range(len(vert)):
        if len(vert[ortant]) != 0:
            vert_x = []
            vert_y = []
            for x in vert[ortant]:
                if len(x) != 0:
                    vert_x.append(x[0])
                    vert_y.append(x[1])

            x_0 = vert_x[0]
            y_0 = vert_y[0]
            vert_x.append(x_0)
            vert_y.append(y_0)

            plt.scatter(vert_x, vert_y, color="#760351", marker=".")
            plt.fill(vert_x, vert_y, linestyle='-', linewidth=1, color="#760351", alpha=0.7)

    for ortant in range(len(vert1)):
        if len(vert1[ortant]) != 0:
            vert1_x = []
            vert1_y = []
            for x in vert1[ortant]:
                if len(x) != 0:
                    vert1_x.append(x[0])
                    vert1_y.append(x[1])
            x_0 = vert1_x[0]
            y_0 = vert1_y[0]
            vert1_x.append(x_0)
            vert1_y.append(y_0)
            plt.scatter(vert1_x, vert1_y, color="#4BBF09", marker=".")
            plt.fill(vert1_x, vert1_y, linestyle='-', linewidth=1, color="#4BBF09", alpha=0.7)

    plt.title(title)
    plt.xlabel("β₀")
    plt.ylabel("β₁")
    plt.legend()

    plt.savefig(f"{fname}")
    plt.show()


def data_corr_naive(Ysint, Ysout, Xi,  ys_int, ys_ext, Xs_lvls, graphics=False):
    y = ip.mid(Ysint)*(1/16384) - 0.5
    epsilon = ip.rad(Ysint)*(1/16384)

    if graphics:
        plot_tol_sys(Xi, Ysint * (1 / 16384) - 0.5, Ysout * (1 / 16384) - 0.5, "tol-before-alg")

    irp_DRSout = ir_problem(ip.inf(Xi), ip.mid(Ysout)*(1/16384) - 0.5, ip.rad(Ysout)*(1/16384))

    tolmax, argmax, env, ccode = tolsolvty(ip.inf(Xi), ip.sup(Xi),
                                           ip.inf(y - epsilon).reshape(-1, 1), ip.sup(y + epsilon).reshape(-1, 1))
    print("\ntolmax: ", tolmax)
    print("\nargmax: ", argmax)
    print("\nenv: ", env)

    if tolmax > 0:
        print("\n!______tolmax > 0______!")

        print("\ntolmax: ", tolmax)
        print("\nargmax: ", argmax)
        print("\nenv: ", env)

        irp_DRSint = ir_problem(ip.inf(Xi), y, epsilon)

        if graphics:
            # ir_plotmodelset([irp_DRSout, irp_DRSint])
            print("I: ", ys_int)
            print("II: ", ys_ext)
            print("III: ", Xs_lvls)
            print_intervals(ys_int, ys_ext, Xs_lvls)
            plt.show()

        b_int = ir_outer(irp_DRSint)
        return b_int, []  # indtoout = None

    print('!______tolmax < 0______!')

    envnegind = np.where(env[:, 1] < 0)[0]
    indtoout = env[envnegind, 0]

    for idx in indtoout:
        idx = int(idx-1)
        y[idx] = mid(Ysout[idx])*(1/16384) - 0.5
        epsilon[idx] = rad(Ysout[idx])*(1/16384)

    if graphics:
        plot_tol_sys(Xi, ip.Interval((y - epsilon), (y + epsilon)), Ysout * (1 / 16384) - 0.5,
                     "tol-after-alg", "Внутренние оценки и внешние")

    tolmax, argmax, env, ccode = tolsolvty(ip.inf(Xi), ip.sup(Xi),
                                           (y - epsilon).reshape(-1, 1), (y + epsilon).reshape(-1, 1))

    print("\ntolmax: ", tolmax)
    print("\nargmax: ", argmax)
    print("\nenv: ", env)

    irp_DRSint = ir_problem(ip.inf(Xi), y, epsilon)


    if graphics:
        ir_plotmodelset([irp_DRSout, irp_DRSint])
        print("I: ", ys_int)
        print("II: ", ys_ext)
        print_intervals(ys_int, ys_ext, Xs_lvls)
        plt.show()

    b_int = ir_outer(irp_DRSint)

    return b_int, indtoout

