# -*- coding: utf-8 -*-

import math
import numpy as np

#Q2 
def euler_solver(func, x0, y, x):
    h = 0.025
    while x0 < x:
        y += h * func(x0, y)
        x0 += h
    print("Approximate solution at x =", x, "is", "%.6f"% y)

#Q2 A
def func(x, y):
    return (x - y + math.cos(x+y))
x0 = 0.1
y0 = 1
x = 1.7
euler_solver(func, x0, y0, x)

#Q2 B
def func(x, y):
    return (x**2 - y**2) / (x**2 + y**2)
x0 = 1
y0 = 0.1
x = 4
euler_solver(func, x0, y0, x)

#Q2 C
def func(x, y):
    return np.exp(x + y) * np.sin(x*y)
x0 = 0.5
y0 = 3
x = 3.8
euler_solver(func, x0, y0, x)

#Q3 A
def func(x, y):
    return x**2 - y + 2*x + 4*y
x0 = 0.3
y0 = -0.18
x = 2.5
euler_solver(func, x0, y0, x)

#Q3 B
def func(x, y):
    return (x + y**2 - 2*x*y) / (x**2 + y**2 - 2*x)
x0 = -1
y0 = 0
x = 2
euler_solver(func, x0, y0, x)

#Q3 C
def func(x, y):
    return np.cos(x+y)*(x**2 + 2*y)
x0 = -2
y0 = -1
x = 1
euler_solver(func, x0, y0, x)

#Q4
def rk_3(func, x0, y0, x):
    h = 0.2
    n = (int)((x - x0) / h)
    y = y0
    for i in range(1, n+1):
        k1 = h * func(x0, y)
        k2 = h * func(x0 + h, y+h*k1)
        k3 = h * func(x0 + 0.5*h, (y+0.5*h*(k2+k1)/2))
        y += (1/6.0) * (k1+k2+4*k3)
        x0 += h
    print("Approximate solution at x =", x, "is", "%.6f"% y)

#Q4 A
def func1(x, y):
    return (x*y) / (x**2+y**2-x*y)
x0 = 0
y0 = 10
x = 3
rk_3(func1, x0, y0, x)

#Q4 B
def func1(x, y):
    return np.tan(x) / np.exp(x+y)
x0 = -0.5
y0 = 1.5
x = 0.75
rk_3(func1, x0, y0, x)

#Q4 C
def func1(x, y):
    return np.cos(x+y)*(x**2+2*y)
x0 = -0.5
y0 = -1.5
x = 0.75
rk_3(func1, x0, y0, x)

#Q5
def rk_4(func, x0, y0, x):
    h = 0.2
    n = (int)((x - x0) / h)
    y = y0
    for i in range(1, n+1):
        k1 = h * func(x0, y)
        k2 = h * func(x0 + 0.5*h, y+0.5*k1*h)
        k3 = h * func(x0 + 0.5*h, y+0.5*k2*h)
        k4 = h * func(x0 + h, y+k3*h)
        y += (1/6.0) * (k1+2*k2+2*k3+k4)
        x0 += h
    print("Approximate solution at x =", x, "is", "%.6f"% y)

#Q5 A
def func1(x, y):
    return x*y+x-y**3
x0 = 1
y0 = -4
x = 2.5
rk_4(func1, x0, y0, x)

#Q5 B
def func1(x, y):
    return np.sin(x) + np.exp(x**2-2*y)
x0 = 2
y0 = 10
x = 3
rk_4(func1, x0, y0, x)

#Q5 C
def func1(x, y):
    return np.cos(x+y)+x**2-x*y
x0 = -2
y0 = -0.5
x = 2
rk_4(func1, x0, y0, x)
