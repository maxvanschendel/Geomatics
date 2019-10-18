import numpy as np
from matplotlib import image


# calculate ndvi
def calculate_ndvi(red, nir):
    with np.errstate(divide='ignore', invalid='ignore'):
        ndvi = np.divide((nir - red), (nir + red))
        ndvi[nir==red] = 0

    return ndvi


# divide ndvi into discrete classes and generate masks of each class
def generate_masks(ndvi, classes):
    masks = {}
    for i in classes.keys():
        masks[i] = np.logical_and(classes[i][0] <= ndvi, ndvi  <= classes[i][1])

    return masks


# generates land use masks from color  and near-infrared images
# using the Normalized Difference Vegetation Index
# global data can be obtained from the sentinel open access hub
if __name__ == '__main__':
    # load images to numpy array
    rgb_image = np.float32(image.imread('.\\satellite_images\\red.tif'))
    infrared_image = np.float32(image.imread('.\\satellite_images\\nir.tif'))

    # extract red and nir channels from images
    red = rgb_image[:, :, 0]
    nir = infrared_image[:, :, 0]

    classes = {'water': [-1.0, 0],
               'barren': [0, 0.2],
               'low_density_vegetation': [0.2, 0.6],
               'high_density_vegetation': [0.6, 1]}

    ndvi = calculate_ndvi(red, nir)
    masks = generate_masks(ndvi, classes)

    # save each mask to black and white image
    for i in classes.keys():
        image.imsave('.\\masks\\' + i + '.tif', masks[i], cmap='Greys')
