# -*- coding: utf-8 -*-

# Q1
#since evenly spaced we use this formula 
def first_diff_beg(y):
    res = (-3*y[0] + 4*y[1] - y[2]) / (2 * (y[1] - y[0]))
    return res
def first_diff_inter(y, ind):
    res = (y[ind + 1] - y[ind - 1]) / (2 * (y[1] - y[0]))
    return res
def second_diff_inter(y, ind):
    res = (y[ind + 1] - 2 * y[ind] + y[ind - 1]) / (2 * (y[1] - y[0]))
    return res

#Q1 A
x = [0, 1, 2, 3]
y = [5, 6, 3, 8]
print("dy/dx at x=0:", first_diff_beg(y))
print("d^2y/dx^2 at x=0:", second_diff_inter(y, 0))

#Q1 B
x = [1.0 , 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]
y = [7.989, 8.403, 8.781, 9.129, 9.451, 9.750, 10.031]
print("dy/dx at x=1.1:", first_diff_inter(y, 1))
print("d^2y/dx^2 at x=1.1:", second_diff_inter(y, 1))
print("dy/dx at x=1.5:", first_diff_inter(y, 5))
print("d^2y/dx^2 at x=1.5:", second_diff_inter(y, 5))

#Q1 C
x = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    y = [30.13, 31.62, 32.87, 33.64, 33.95, 33.81, 33.24]
print("dy/dx at x=0.3:", first_diff_inter(y, 3))
print("d^2y/dx^2 at x=0.3:", second_diff_inter(y, 3))

#Q2
def trape_rule_pts(y, b, a):
    res = y[0] + y[-1]
    for f in y:
        res += 2 * f
    res *= (b - a) / (2 * len(y))
    return res
def trape_rule(f, b, a, n):
    h = (b - a ) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += 2 * f(a + i*h)
    return (h / 2) * s
def simp_13_rule(f, b, a, n):
    h = (b - a) / n
    res = 0
    for i in range(n+1):
        if i == 0 or i == n:
            res += f(a + i * h)
        elif i % 2 != 0:
            res += 4 * f(a + i * h)
        else:
            res += 2 * f(a + i * h)
    return res * (h / 3)
def simp_38_rule(f, b, a, n):
    h = (b - a) / n
    res = f(a) + f(b)
    for i in range(n):
        if i % 3 == 0:
            res += 2 * f(a + i*h)
        else:
            res += 3 * f(a + i*h)
    return (3*h / 8) *res

#Q2 A
x = [7.47, 7.49, 7.50, 7.51, 7.52]
y = [1.93, 1.95, 1.98, 2.01, 2.03, 2.06]
print("Area under {} and {}=".format(x[0], x[-1]), trape_rule_pts(y, x[-1], x[0]))

#Q2 B
x = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
y = [8, 17, 24, 28, 30, 20, 12, 6, 2, 0]
print("Area under {} and {}=".format(x[0], x[-1]), trape_rule_pts(y, x[-1], x[0]))

import math

#Q2 C
def f(x):
    return 1 / (1 + math.log(x))
b = 1.3
a = 0.3
n = 8
print("Area using Trapezoidal Rule: ", trape_rule(f, b, a, n))
print("Area using Simpsons 1/3 Rule: ", simp_13_rule(f, b, a, n))
print("Area using Simpsons 3/8 Rule: ", simp_38_rule(f, b, a, n))

#Q2 D
f = lambda x: math.sin(x)
b = math.pi / 2
a = 0
n = 6
print("Area using Trapezoidal Rule: ", trape_rule(f, b, a, n))
print("Area using Simpsons 1/3 Rule: ", simp_13_rule(f, b, a, n))
print("Area using Simpsons 3/8 Rule: ", simp_38_rule(f, b, a, n))
