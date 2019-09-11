# GEO1000 - Assignment 1
# Authors: 
# Studentnumbers:

import math

def distance(x1, y1, x2, y2):
    return math.sqrt(((x2 - x1)**2) + ((y2 - y1)**2))

def angle(a, b, c):
    return math.acos((a**2+b**2-c**2)/(2*a*b))

def cot(x):
    return 1/math.tan(x)

def tienstra(ax, ay, bx, by, cx, cy, alpha, beta):
    alpha_rad = math.radians(alpha)
    beta_rad = math.radians(beta)
    gamma_rad = 2*math.pi - alpha_rad - beta_rad

    AB = distance(ax, ay, bx, by)
    BC = distance(bx, by, cx, cy)
    CA = distance(ax, ay, cx, cy)

    angle_A = angle(AB, CA, BC)
    angle_B = angle(AB, BC, CA)
    angle_C = angle(CA, BC, AB)

    K_1 = 1 / (cot(angle_A) - cot(alpha_rad))
    K_2 = 1 / (cot(angle_B) - cot(beta_rad))
    K_3 = 1 / (cot(angle_C) - cot(gamma_rad))

    px = ((K_1 * ax) + (K_2 * bx) + (K_3 * cx))/(K_1 + K_2 + K_3)
    py = ((K_1 * ay) + (K_2 * by) + (K_3 * cy)) / (K_1 + K_2 + K_3)

    print(px, py)

tienstra(
    1000.0, 5300.0,
    2200.0, 6300.0,
    3100.0, 5000.0,
    115.052, 109.3045)

