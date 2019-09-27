# calculates mean value of list
def mean(a):
    return sum(a) / len(a)


# multiplies two vectors together
def multiply_vectors(a, b):
    out = []
    for i in enumerate(a):
        out.append(i[1] * b[i[0]])

    return out


# least squares adjustment
def least_squares(x, y):
    n = len(x)
    m_x, m_y = mean(x), mean(y)

    ss_xy = sum(multiply_vectors(x, y)) - n * m_y * m_x
    ss_xx = sum(multiply_vectors(x, x)) - n * m_x * m_x

    b_1 = ss_xy / ss_xx
    b_0 = m_y - b_1 * m_x

    return b_0, b_1


# measurements
D = [-29, -21, -9, 0, 11, 19, 30]
clay_percent = [.07, .23, .41, .53, .69, .82, .99]

# estimating coefficients
parameters = least_squares(D, clay_percent)
print("a =", parameters[0])
print("b =", parameters[1])



