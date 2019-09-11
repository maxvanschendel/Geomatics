from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


class Kernel:
    def __init__(self):
        self.operation = np.array([[1, 1, 1],
                                   [0,0,0],
                                   [-1,-1,-1]])

        self.x_buf = int((self.operation.shape[0]-1) / 2)
        self.y_buf = int((self.operation.shape[1]-1) / 2)
        self.pos = [self.x_buf, self.y_buf]

    def move_col(self):
        self.pos[0] += 1

    def move_row(self):
        self.pos[0] = self.x_buf
        self.pos[1] += 1

    def apply_operation(self, values):
        return (values * self.operation).sum()


def load_image(fn):
    img = Image.open(fn).convert('L').resize((125, 125))
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


def remap_val(n, in_min, in_max, out_min, out_max):
    return out_min + (((n - in_min) / (in_max-in_min))*(out_max-out_min))


def remap_array(ar):
    vfunc = np.vectorize(remap_val)
    return vfunc(ar, ar.min(), ar.max(), 0, 255)


def run_conv_filter(input_file):
    img = load_image(input_file)
    img_shape = img.shape
    output_image = np.zeros(img_shape, dtype=np.float32)

    k = Kernel()
    while k.pos[0] < img_shape[0] - k.x_buf and k.pos[1] < img_shape[1] - k.y_buf:
        kernel_values = img[k.pos[0] - k.x_buf:k.pos[0] + k.x_buf + 1, k.pos[1] - k.y_buf: k.pos[1] + k.y_buf + 1]
        output_image[k.pos[0]][k.pos[1]] = k.apply_operation(kernel_values)

        k.move_col()
        if k.pos[0] == img_shape[0] - k.x_buf:
            k.move_row()

        print(k.pos)

    output_remap = remap_array(output_image)
    print(output_remap)
    plt.imshow(output_remap, interpolation='nearest', cmap='gray')
    plt.show()


if __name__ == '__main__':
    run_conv_filter('./apple.jpg')