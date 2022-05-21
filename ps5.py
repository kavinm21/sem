# -*- coding: utf-8 -*-

import numpy as np

# Q1
def linear_inter(fx0, fx1, x0, x1, x):
    return fx0 + (((fx1 - fx0) / (x1 - x0)) * (x - x0))

# Q1 A
x0 = -1
fx0 = -8
x1 = 2
fx1 = 1
x = 0
fx = linear_inter(fx0, fx1, x0, x1, x)
print("f({}) =".format(x), fx)

# Q1 B
x0 = 5
fx0 = 12
x1 = 9
fx1 = 15
x = 6
fx = linear_inter(fx0, fx1, x0, x1, x)
print("f({}) =".format(x), fx)

# Q2
def quadratic_inter(fx0, fx1, fx2, x0, x1, x2, x):
    b0 = fx0
    b1 = (fx1 - fx0) / (x1 - x0)
    b2 = (((fx2 - fx1) / (x2 - x1)) - b1) / (x2 - x0)
    return b0 + (b1 * (x - x0)) + (b2 * (x - x0) * (x - x1))

# Q2 A
x0 = 0
fx0 = 659
x1 = 2
fx1 = 705
x2 = 3
fx2 = 729
x = 2.75
fx = quadratic_inter(fx0, fx1, fx2, x0, x1, x2, x)
print("f({}) =".format(x), fx)

# Q2 B
x0 = 93
fx0 = 11.38
x1 = 96.2
fx1 = 12.80
x2 = 100
fx2 = 14.7
x = 2.75
fx = quadratic_inter(fx0, fx1, fx2, x0, x1, x2, x)
print("f({}) =".format(x), fx)

# Q3
def divided_diff_table(x, y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = ((y[j][i - 1] - y[j + 1][i - 1]) / (x[j] - x[i + j]))
    return y
def product_term(i, val, x):
    pro = 1
    for j in range(i):
        pro *= val - x[j]
    return pro
def divided_diff(val, x, y, n):
    sum = y[0][0]
    for i in range(1, n):
        sum += product_term(i, val, x) * y[0][i]
    return sum

# Q3 A
n = 4
y = [[0 for i in range(10)] for j in range(10)]
x = [654, 658, 659, 661]
y[0][0] = 2.8156
y[1][0] = 2.8182
y[2][0] = 2.8189
y[3][0] = 2.8202
y = divided_diff_table(x, y, n)
val = 656
print("f({}) =".format(val), divided_diff(val, x, y,n))

# Q3 B
n = 5
y = [[0 for i in range(10)] for j in range(10)]
x = [0, 0.1, 0.2, 0.3, 0.4]
y[0][0] = 1
y[1][0] = 1.1052
y[2][0] = 1.2214
y[3][0] = 1.3499
y[4][0] = 2.8202
y = divided_diff_table(x, y, n)
val = 0.38
print("f({}) =".format(val), divided_diff(val, x, y,n))

# Q4 A
n = 5
x = [45, 50, 55, 60, 65]
y = [[0 for i in range(10)] for j in range(10)]
y[0][0] = 114.84
y[1][0] = 96.16
y[2][0] = 83.32
y[3][0] = 74.48
y[4][0] = 68.48
y = divided_diff_table(x, y, n)
val = 46
print("f({}) =".format(val), divided_diff(val, x, y,n))

# Q4 B
n = 5
x = [45, 50, 55, 60, 65]
y = [[0 for i in range(10)] for j in range(10)]
y[0][0] = 114.84
y[1][0] = 96.16
y[2][0] = 83.32
y[3][0] = 74.48
y[4][0] = 68.48
y = divided_diff_table(x, y, n)
val = 63
print("f({}) =".format(val), divided_diff(val, x, y,n))

# Q5
def forward_diff_table(y, n):
    for i in range(1, n):
        for j in range(n - i):
            y[j][i] = y[j + 1][i - 1] - y[j][i - 1]
    return y
def calc_u(u, n, f=1):
    temp = u
    for i in range(1, n):
        temp *= u - i
    return temp
def get_factorial(n):
    f = 1
    for i in range(2, n + 1):
        f *= i
    return f
def forward_diff(val, x, y, n):
    sum = y[0][0]
    u = (val - x[0]) / (x[1] - x[0])
    for i in range(1, n):
        sum += (calc_u(u, i) * y[0][i]) / get_factorial(i)
    return sum
def u_cal(u, n):
	temp = u;
	for i in range(1, n):
		temp = temp * (u + i);
	return temp
def backward_diff(value, x, y, n):
    for i in range(1, n):
        for j in range(n - 1, i+1, -1):  
            y[j][i] = y[j][i - 1] - y[j - 1][i - 1]
    sum = y[n - 1][0];
    u = (value - x[n - 1]) / (x[1] - x[0]);
    for i in range(1, n):
        sum = sum + (u_cal(u, i) * y[n - 1][i]) / get_factorial(i);  
    return y, sum

# Q5 A
n = 7
x = [i for i in range(3, 10)]
y = [[0 for i in range(n)] for j in range(n)]
y[0][0] = 4.8
y[1][0] = 8.4
y[2][0] = 14.5
y[3][0] = 23.6
y[4][0] = 36.2
y[5][0] = 52.8
y[6][0] = 73.9
y1 = forward_diff_table(y, n)
val = 3.5
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)
val = 8.5
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)

# Q5 B
n = 5
x = [45, 50, 55, 60,65]
y = [[0 for i in range(n)] for j in range(n)]
y[0][0] = 114.8
y[1][0] = 96.16
y[2][0] = 83.32
y[3][0] = 74.48
y[4][0] = 68.48
y1 = forward_diff_table(y, n)
val = 47
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)
val = 64
print("Newton's Forward Difference - f({}) =".format(val), forward_diff(val, x, y1,n))
y2, res = backward_diff(val, x, y,n)
print("Newton's Backward Difference - f({}) =".format(val), res)

# Q5
def lagrange_interpolate(f, x1):
    res = 0.0
    n = len(f)
    for i in range(n):
        term = f[i][1]
        for j in range(n):
            if j != i:
                term *= (x1 - f[j][0]) / (f[i][0] - f[j][0])
        res += term
    return res

# Q5 A
f = [[5, 12], [6, 13], [9, 14], [11, 16]]
val = 10
print("Lagrange's Interpolation: f({}) =".format(val), lagrange_interpolate(f, val))
