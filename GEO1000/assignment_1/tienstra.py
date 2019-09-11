# GEO1000 - Assignment 1
# Authors: Max van Schendel
# Studentnumbers: 4384644

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

    ab = distance(ax, ay, bx, by)
    bc = distance(bx, by, cx, cy)
    ca = distance(ax, ay, cx, cy)

    angle_a = angle(ab, ca, bc)
    angle_b = angle(ab, bc, ca)
    angle_c = angle(ca, bc, ab)

    k_1 = 1 / (cot(angle_a) - cot(alpha_rad))
    k_2 = 1 / (cot(angle_b) - cot(beta_rad))
    k_3 = 1 / (cot(angle_c) - cot(gamma_rad))

    px = ((k_1 * ax) + (k_2 * bx) + (k_3 * cx))/(k_1 + k_2 + k_3)
    py = ((k_1 * ay) + (k_2 * by) + (k_3 * cy)) / (k_1 + k_2 + k_3)

    print((px, py))

tienstra(
    1000.0, 5300.0,
    2200.0, 6300.0,
    3100.0, 5000.0,
    115.052, 109.3045)

tienstra(
    0, 0,
    0, 10,
    10, 5,
    120, 120)


