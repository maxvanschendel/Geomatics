from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


# kernel object
class Kernel:
    def __init__(self):
        self.operation = np.array([[1, 2, 1],
                                   [2, 4, 2],
                                   [1, 2, 1]])

        self.x_buf = self.operation.shape[0] - 2
        self.y_buf = self.operation.shape[1] - 2

        self.pos = [self.x_buf, self.y_buf]

    def move_col(self):
        self.pos[0] += 1

    def move_row(self):
        self.pos[0] = self.x_buf
        self.pos[1] += 1

    def apply_operation(self, values):
        return (values * self.operation / 16).sum()


# load image to numpy array
def load_image(fn):
    img = Image.open(fn).convert('L').resize((50, 50))
    img.load()
    data = np.asarray( img, dtype="int32")
    return data


# move kernel across image, applying filter at every step
# outputs to new image
def run_conv_filter(input_file):

    shape = img.shape

    k = Kernel()
    output_image = np.zeros(img.shape)

    while k.pos[0] < shape[0] - k.x_buf and k.pos[1] < shape[1] - k.y_buf:
        kernel_values = img[k.pos[0] - k.x_buf:k.pos[0] + k.x_buf + 1, k.pos[1] - k.y_buf: k.pos[1] + k.y_buf + 1]
        output_image[k.pos[0]][k.pos[1]] = k.apply_operation(kernel_values)

        k.move_col()
        if k.pos[0] == shape[0] - k.x_buf:
            k.move_row()

    plt.imshow(output_image, interpolation='nearest', cmap='gray')
    plt.show()


img = load_image(input_file)

run_conv_filter('./apple.jpg')
