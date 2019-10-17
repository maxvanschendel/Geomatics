import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


# load image to numpy array
def load_image(fn):
    img = Image.open(fn)
    img.load()
    data = np.asarray(img)
    return data


def plot(ar):
    plt.imshow(ar)
    plt.show()


def calculate_ndvi(red, nir):
    return (nir - red) / (nir + red)


def is_between(x, a, b):
    if b >= x >= a:
        return True
    else:
        return False


def classify(val):
    classes = {'built': [-1.0, -.1],
               'water': [-.1, 0.1],
               'low density vegetation': [0.1, 0.4],
               'high density vegetation': [0.4, 1]}

    ranges = classes.values()
    for i in enumerate(ranges):
        if is_between(val, i[1][0], i[1][1]):
            return i[0]


rgb_image = load_image('C:\\Users\\Max\\PycharmProjects\\personal_projects\\Geomatics\\GEO1001\\satellite_images\\rgb.tif')
red = rgb_image[:, :, 0]

infrared_image = load_image('C:\\Users\\Max\\PycharmProjects\\personal_projects\\Geomatics\\GEO1001\\satellite_images\\nir.tif')
nir = infrared_image[:, :, 0]

ndvi_vector = np.vectorize(calculate_ndvi)
ndvi = ndvi_vector(red, nir)

classify_vector = np.vectorize(classify)
classified_ndvi = classify_vector(ndvi)

plot(classified_ndvi)
