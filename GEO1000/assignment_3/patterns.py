# GEO1000 - Assignment 3
# Authors:
# Studentnumbers:


def wkt(p1, p2, p3, p4):
    """Returns Well Known Text string of 
    square which is defined by 4 points.

    Arguments:

      p1--p4: 2-tuple of floats, the 4 corners of the square

    Returns:

      str (WKT of square) - POLYGON((x1 y1, x2 y2, x3 y3, x4 y4, x1 y1))

      note:
          - Order of coordinates (counterclockwise):
              p1 = bottom left corner,
              p2 = bottom right corner,
              p3 = top right corner,
              p4 = top left corner
          - Bottom left corner has to be repeated
          - Coordinates should be output with *6* digits behind the dot.
    """
    return "POLYGON (({} {}, {} {}, {} {}, {} {}, {} {}))".format(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1], p4[0], p4[1], p1[0], p1[1])


def pattern_a(l, c, size, ratio, file_nm):
    """Draw a pattern of squares.

    Arguments:
        l - level to draw
        c - 2-tuple of floats (the coordinates of center of the square)
        size - half side length of the square
        ratio - how much smaller the next square will be drawn
        file_nm - file name of file to write to

    Returns:
        None
    """

    if l == 0:
        return None
    else:

        p1 = (c[0] - size, c[1] - size)
        p2 = (c[0] + size, c[1] - size)
        p3 = (c[0] + size, c[1] + size)
        p4 = (c[0] - size, c[1] + size)

        with open(file_nm, 'a') as out_file:
            text = wkt(p1, p2, p3, p4)
            out_file.write(str(text) + '\n')

        return [pattern_a(l - 1, p1, size/ratio, ratio, file_nm),
                pattern_a(l - 1, p2, size / ratio, ratio, file_nm),
                pattern_a(l - 1, p3, size / ratio, ratio, file_nm),
                pattern_a(l - 1, p4, size / ratio, ratio, file_nm)]


def pattern_b(l, c, size, ratio, file_nm):
    """Draw a pattern of squares.

    Arguments:
        l - level to draw
        c - 2-tuple of floats (the coordinates of center of the square)
        size - half side length of the square
        ratio - how much smaller the next square will be drawn
        file_nm - file name of file to write to

    Returns:
        None
    """

    p1 = (c[0] - size, c[1] - size)
    p2 = (c[0] + size, c[1] - size)
    p3 = (c[0] + size, c[1] + size)
    p4 = (c[0] - size, c[1] + size)
    text = wkt(p1, p2, p3, p4)

    if l == 0:
        return None

    else:
        with open('./'+file_nm, 'r+') as out_file:
            lines = out_file.read().splitlines()
            lines = list(reversed(lines))
            lines.append(text)

            with open(file_nm, 'w') as out_file:
                for i in reversed(lines):
                    out_file.write(i + '\n')

        return [pattern_b(l - 1, p1, size / ratio, ratio, file_nm),
                pattern_b(l - 1, p2, size / ratio, ratio, file_nm),
                pattern_b(l - 1, p3, size / ratio, ratio, file_nm),
                pattern_b(l - 1, p4, size / ratio, ratio, file_nm)]


def pattern_c(l, c, size, ratio, file_nm):
    """Draw a pattern of squares.

    Arguments:
        l - level to draw
        c - 2-tuple of floats (the coordinates of center of the square)
        size - half side length of the square
        ratio - how much smaller the next square will be drawn
        file_nm - file name of file to write to

    Returns:
        None
    """

    p1, p2, p3, p4 = (c[0] - size, c[1] - size), (c[0] + size, c[1] - size), \
                     (c[0] + size, c[1] + size), (c[0] - size, c[1] + size)

    if l == 0:
        return None

    else:
        file = open(file_nm, 'a')
        file.write(wkt(p1, p2, p3, p4) + '\n')

        for p in [p3, p4]:
            pattern_c(l-1, p, size/ratio, ratio, file_nm)

        for p in [p1, p2]:
            file.close()
            pattern_c(l-1, p, size/ratio, ratio, file_nm)


def main(n=5, c=(0.0, 0.0), size=10.0, ratio=2.2):
    """The starting point of this program. 
    Writes for every output file the first line and allows
    to influence how the resulting set of squares look.

    Arguments:
        n - levels of squares that are produced
        c - coordinate of center of the drawing
        size - half side length of the first square
        ratio - how much smaller a square on the next level will be drawn

    """
    funcs = [pattern_a, pattern_b, pattern_c]
    file_nms = ['pattern_a.txt', 'pattern_b.txt', 'pattern_c.txt']

    for func, file_nm_out in zip(funcs, file_nms):
        with open(file_nm_out, 'w+') as text_file:
            pass
        func(n, c, size, ratio, file_nm_out)

        with open(file_nm_out, 'r+') as text_file:
            lines = list(reversed(text_file.readlines()))
            lines.append('geometry\n')
            lines = list(reversed(lines))

        with open(file_nm_out, 'w') as text_file:
            for i in lines:
                text_file.write(i)


if __name__ == "__main__":
    main()
