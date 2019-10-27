import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from StyleFrame import StyleFrame, utils


# read excel file to usable numpy arrays
def load_multispectral_data(excel_file):
    df = pd.read_excel(excel_file, 'Multispectral Image')

    nir = df[0:20].iloc[:, 1:].to_numpy()
    red = df[22:42].iloc[:, 1:].to_numpy()

    return nir, red


def load_training_data(excel_file):
    sf = StyleFrame.read_excel(excel_file, sheet_name='Training Samples', read_style=True, use_openpyxl_styles=False)
    return StyleFrame(sf.applymap(get_classes_from_colors)).data_df[0:20].iloc[:, 1:].to_numpy(dtype=np.str)


def get_classes_from_colors(cell):
    if cell.style.bg_color in {utils.colors.green, 'FF92D050'}:
        return 'vegetation'
    elif cell.style.bg_color in {utils.colors.red, 'FFFF0000'}:
        return 'bare ground'
    elif cell.style.bg_color in {utils.colors.blue, '3275c8'}:
        return 'water'
    else:
        return 'unclassified'


# calculate ndvi and set invalid values (zero division) to zero
def calculate_ndvi(nir, red):
    ndvi = np.divide(nir - red, nir + red)
    return np.nan_to_num(ndvi)


def plot_histogram(ndvi, interval, filename):
    histo = []
    series = np.arange(-1, 1, interval)

    for i in series:
        in_interval = np.logical_and(i <= ndvi, ndvi < i + interval)
        histo.append(in_interval.sum())

    plt.bar(series, histo, width=interval, align='edge', edgecolor='white', color='grey')

    plt.title('Histogram of NDVI values')
    plt.xlabel('Range of NDVI values')
    plt.ylabel('Amount of values within range')

    plt.savefig(filename)


def plot_scatter(red, nir, colors='Grey'):
    plt.scatter(red, nir, c=colors)

    plt.title('Relationship of red and near-infrared channels')
    plt.xlabel('Red channel')
    plt.ylabel('Near-infrared channel')


def mv_cov_covinv(ar1, ar2):
    obs_vectors = np.ma.vstack((ar1, ar2)).T
    mean_vector = np.mean(obs_vectors, axis=0)
    covariance_matrix = np.cov(obs_vectors)
    covariance_matrix_inv = np.linalg.pinv(covariance_matrix)

    return {'mean vector': mean_vector,
            'covariance matrix': covariance_matrix,
            'inverse covariance matrix': covariance_matrix_inv}


def minimum_distance_to_mean(vec, means):
    distances = [np.linalg.norm(vec-i) for i in means]
    min_distance = distances[np.argmin(distances)]
    return (np.argmin(distances) + 1, min_distance)


if __name__ == '__main__':
    ### assignment 1
    fn = './Multispectral Classification.xlsx'
    nir, red = load_multispectral_data(fn)[0], load_multispectral_data(fn)[1]
    ndvi = calculate_ndvi(nir, red)

    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(111)
    plt.imshow(ndvi)
    plt.colorbar(orientation='vertical')
    plt.title('NDVI values of 20x20 area')
    plot_histogram(ndvi, 0.2, 'histogram.jpg')

    ### assignment 2
    plot_scatter(red, nir)
    plt.savefig('scatter.jpg')
    plt.close()

    ### assignment 3
    training_classes = load_training_data(fn)

    # get masks for each class
    water_mask = np.isin(training_classes, 'water', invert=True)
    bg_mask = np.isin(training_classes, 'bare ground', invert=True)
    veg_mask = np.isin(training_classes, 'vegetation', invert=True)
    unc_mask = np.isin(training_classes, 'unclassified', invert=True)

    # plot each class with a different color
    plot_scatter(red[~unc_mask], nir[~unc_mask], colors='lightgrey')
    plot_scatter(red[~water_mask], nir[~water_mask], colors='blue')
    plot_scatter(red[~bg_mask], nir[~bg_mask], colors='red')
    plot_scatter(red[~veg_mask], nir[~veg_mask], colors='green')
    plt.savefig('scatter_f.jpg')
    plt.close()

    ### assignment 5
    # compute mean vector, covariance matrix and inverse of covariance matrix
    water_stats = mv_cov_covinv(red[~water_mask], nir[~water_mask])
    bg_stats = mv_cov_covinv(red[~bg_mask], nir[~bg_mask])
    veg_stats = mv_cov_covinv(red[~veg_mask], nir[~veg_mask])

    ### assignment 6
    obs_vecs = np.array((red, nir)).T
    means = (veg_stats['mean vector'], bg_stats['mean vector'], water_stats['mean vector'])

    classified = np.zeros(ndvi.shape)
    distances = np.zeros(ndvi.shape)
    for i in range(len(obs_vecs)):
        for j in range(len(obs_vecs[i])):
            pixel_class = minimum_distance_to_mean(obs_vecs[i][j], means)
            classified[j][i] = pixel_class[0]
            distances[j][i] = pixel_class[1]

    # threshold distance values
    classified[distances > 2*np.std(distances)] = None

    # write to excel file
    df = pd.DataFrame(classified)
    df.to_excel('classified.xlsx', index=False)
    plt.imshow(classified)
    plt.show()

    colors = {0:'lightgrey', 1: 'green', 2: 'red', 3:'blue'}

    # assignment 7
    for i in range(0, 4):
        plt.scatter(red[classified == i], nir[classified == i], c=colors[i])
        in_class = ndvi[classified == i]

        in_class[abs(in_class - np.mean(in_class)) > np.std(in_class)] = None
        ndvi_range = (np.nanmin(in_class), np.nanmax(in_class))

    plt.show()













