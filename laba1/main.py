import numpy as np
np.float_ = np.float64
from intvalpy import Interval, mid, subset, intersection
from tabulate import tabulate


def print_table(matrix, headers):
    print(tabulate(matrix, headers, tablefmt="simple_grid", stralign='center'))


def CreateMatrix(eps):
    A = Interval([
          [[1.05-eps, 1.05+eps], [0.95-eps, 0.95+eps]],
          [[1-eps, 1+eps], [1-eps, 1+eps]]
        ])

    output = [
        [str(A[0][0]), str(A[0][1])],
        [str(A[1][0]), str(A[1][1])]
    ]

    print(f"\n---------------------------------------------------------\neps = {eps}")
    print_table(output, headers=[])

    return A


def Determinant(A):
    a = A[0][0]*A[1][1]
    b = A[0][1]*A[1][0]
    res = a - b
    print(f"\n1) A[0][0]*A[1][1] = {A[0][0]}*{A[1][1]} = {a}")
    print(f"\n2) A[0][1]*A[1][0] = {A[0][1]}*{A[1][0]} = {b}")
    print(f"\n3) A[0][0]*A[1][1] - A[0][1]*A[1][0] = {a}*{b} = {res}")
    return res


a = 3
eps_start = 0
step = 0.01
print("STEP = ", step)

A = CreateMatrix(eps_start)
interval = Determinant(A)

iter_count = 0
count_enc = 0
count_dec = 0

while 0 not in interval:
    count_enc += 1
    print("\ndeterminant = ", interval)
    eps = eps_start + step*count_enc
    A = CreateMatrix(eps)
    interval = Determinant(A)
    iter_count += 1

eps_start = eps
step = 0.0001
print("STEP = ", step)

while 0 in interval:
    count_dec += 1
    print("\ndeterminant = ", interval)
    eps = eps_start - step*count_dec
    A = CreateMatrix(eps)
    interval = Determinant(A)
    iter_count += 1

print("iter_count", iter_count)
eps_min = round(eps+step, a)
A = CreateMatrix(eps_min)
interval = Determinant(A)
print("\nMinimal eps = ", eps_min)

eps_max = chr(4113)
print(f"\ndiapason eps = [{eps_min}, +{eps_max}]")

output = [
    [str(A[0][0]), str(A[0][1])],
    [str(A[1][0]), str(A[1][1])]
]
print_table(output, headers=[])

print(f"\ndet(A) for minimal eps = {interval}")

def CutInt(A):
    if intersection(A[0][0], A[1][0]):
        if intersection(A[1][1], A[0][1]):
            res = [
                [str(intersection(A[0][0], A[1][0])), str(intersection(A[1][1], A[0][1]))],
                [str(intersection(A[0][0], A[1][0])), str(intersection(A[1][1], A[0][1]))]
            ]
            return res
    if intersection(A[0][0], A[0][1]):
        if intersection(A[1][1], A[1][0]):
            res = [
                [intersection(A[0][0], A[0][1]), intersection(A[0][0], A[0][1])],
                [intersection(A[1][1], A[1][0]), intersection(A[1][1], A[1][0])]
            ]
            return res
    return None


new_A = CutInt(A)
if new_A:
    print("\nПересечение не пусто => вырожденная точечная матрица А' существует: ")
    print_table(new_A, headers=[])
else:
    print("\nПересечение пусто => вырожденной точечной матрицы А' для найденного значения epsilon не существует")
