# GEO1000 - Assignment 2
# Authors: Max van Schendel
# Studentnumbers: 4384644


def move(start, end, moves):
    if moves == 0:
        if start == end:
            return 1
        else:
            return 0
    else:
        return move(start + 1, end, moves - 1) + move(start - 1, end, moves - 1)


if __name__ == "__main__":
    print(move(1, 4, 5))
