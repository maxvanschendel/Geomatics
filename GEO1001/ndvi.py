import numpy as np
from PIL import Image
from matplotlib import pyplot as plt



# load image to numpy array
def load_image(fn):
    img = Image.open(fn)
    img.load()
    data = np.asarray(img)

    return data


# displays numpy array as image
def plot(ar, cmap):
    plt.imshow(ar, cmap)
    plt.show()


# displays numpy array as image
def save(ar, cmap, fn):
    plt.imshow(ar, cmap)
    plt.savefig(fn)


# calculate normalized difference vegetation index (ndvi)
def calculate_ndvi(red, nir):
    try:
        return (nir - red) / (nir + red)
    except ZeroDivisionError:
        return 0


# checks if x is between a and b
def is_between(x, a, b):
    if b >= x >= a:
        return True
    else:
        return False


# finds class which a value belongs to based on ranges
def classify(val):
    classes = {'built': [-1.0, -.1],
               'water': [-.1, 0.1],
               'low density vegetation': [0.1, 0.4],
               'high density vegetation': [0.4, 1]}

    ranges = classes.values()
    for i in enumerate(ranges):
        if is_between(val, i[1][0], i[1][1]):
            return i[0]


# convert continuous ndvi value to discrete classes
def classify_image(red_image, nir_image):
    ndvi_vector = np.vectorize(calculate_ndvi)
    ndvi = ndvi_vector(red_image, nir_image)
    classify_vector = np.vectorize(classify)

    return classify_vector(ndvi)


# generate black and white images for each class
def generate_masks(classified):
    classes = {'built': [-1.0, -.1],
               'water': [-.1, 0.1],
               'low density vegetation': [0.1, 0.4],
               'high density vegetation': [0.4, 1]}

    classes_keys, classes_vals = list(classes.keys()), classes.values()

    masks = {}
    for i in range(len(classes_vals)):
        masks[classes_keys[i]] = classified == i

    return masks


# generates land use masks from color  and near-infrared images
# using the Normalized Difference Vegetation Index
# global data can be obtained from SentinelHub
if __name__ == '__main__':
    # load images to numpy array
    rgb_image = load_image('.\\satellite_images\\sentinel2_red.tif')
    infrared_image = load_image('.\\satellite_images\\sentinel2_nir.tif')

    # extract red and nir channels from images
    red = rgb_image#[:, :, 0]
    nir = infrared_image#[:, :, 0]

    # classify land use and generate masks for each class
    classified_ndvi = classify_image(red, nir)
    landuse_masks = generate_masks(classified_ndvi)

    # display mask
    plot(classified_ndvi, cmap='viridis')
    plot(landuse_masks['built'], cmap='Greys')
