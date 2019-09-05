# GEO1000 - Assignment 1
# Authors:
# Studentnumbers:

from math import sqrt


def abc(a, b, c):
    D = discriminant(a, b, c)
    print("The roots of %2.1f x^2 + %2.1f x + %2.1f are:" % (a, b, c))

    if D > 0:
        x_positive, x_negative = (-b + sqrt(D))/(2*a), (-b - sqrt(D))/(2*a)
        print("x1 = %2.1f, x2 = %2.1f" % (x_positive, x_negative))

    elif D < 0:
        print("not real")

    else:
        x = -b / (2*a)
        print("x = %2.1f" % x)


def discriminant(a, b, c):
    return b**2 - (4*a*c)


abc(2.0, 0.0, 0.0)
abc(1.0, -4., -5.)
abc(3.0, 4.5, 9.0)
