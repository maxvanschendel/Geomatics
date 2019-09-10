from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


class Kernel:
    def __init__(self):
        self.operation = np.array([[1, 1, 1],
                                   [0, 0, 0],
                                   [-1, -1, -1]])

        self.x_buf = self.operation.shape[0] - 2
        self.y_buf = self.operation.shape[1] - 2

        self.pos = [self.x_buf, self.y_buf]

    def move_col(self):
        self.pos[0] += 1

    def move_row(self):
        self.pos[0] = self.x_buf
        self.pos[1] += 1

    def apply_operation(self, values):
        return (values * self.operation).sum()


def load_image(fn) :
    img = Image.open(fn).convert('L').resize((250, 250))
    img.load()
    data = np.asarray( img, dtype="int32")
    return data


def run_conv_filter(input_file):
    img = load_image(input_file)
    shape = img.shape

    k = Kernel()
    output_image = np.zeros(img.shape)

    while k.pos[0] < shape[0] - k.x_buf and k.pos[1] < shape[1] - k.y_buf:
        kernel_values = img[k.pos[0] - k.x_buf:k.pos[0] + k.x_buf + 1, k.pos[1] - k.y_buf: k.pos[1] + k.y_buf + 1]
        output_image[k.pos[0]][k.pos[1]] = k.apply_operation(kernel_values)

        k.move_col()
        if k.pos[0] == shape[0] - k.x_buf:
            k.move_row()

        print(k.pos)

    plt.imshow(output_image, interpolation='nearest', cmap='gray')
    plt.show()


run_conv_filter('./apple.jpg')