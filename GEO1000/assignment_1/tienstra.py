# GEO1000 - Assignment 1
# Authors: 
# Studentnumbers:

import math

def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

def angle(a, b, c):
    return math.acos((a**2+b**2+c**2)/(2*a*b))

def cot(x):
    pass

def tienstra(ax, ay, bx, by, cx, cy, alpha, beta):
    pass

tienstra(
    1000.0, 5300.0,
    2200.0, 6300.0,
    3100.0, 5000.0,
    115.052, 109.3045)

