import matplotlib.pyplot as plt
import statistics as stats


def combine_data(files, newfile_name):
    with open(newfile_name, 'a') as nf:
        for i in files:
            with open(i) as source_file:
                nf.write(source_file.read())


def position_fixes(file):
    with open(file) as open_file:
        data = [i.split(',') for i in open_file.readlines()]

    gga = list(filter(lambda x: x[0] == '$GPGGA' and x[6] == '1', data))
    lat, lon, alt = [float(i[2]) for i in gga], [float(i[4]) for i in gga], [float(i[9]) for i in gga]

    return lat, lon, alt


if __name__ == '__main__':
    # unoccluded and occluded data files
    unoc = './data/unoccluded.txt'
    oc = './data/occluded.txt'

    # combine seperate files into one dataset
    combine_data(['./data/a.txt', './data/b.txt', './data/c.txt'], unoc)
    combine_data(['./data/a2.txt', './data/b2.txt', './data/c2.txt'], oc)

    # extract GGA latitude and longitude data
    unoc_pos = position_fixes(unoc)
    oc_pos = position_fixes(oc)

    # create scatterplot of data
    # occluded data
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(oc_pos[1], oc_pos[0], s=2, c='green')

    plt.scatter(stats.mean(oc_pos[1]), stats.mean(oc_pos[0]), c='red', marker='x')
    plt.title('Occluded GPS position fixes', y=1.05)
    plt.xlabel('Longitude', x=0.4)
    plt.ylabel('Latitude')
    plt.legend(['Position', 'Mean'])

    # unoccluded data
    ax2 = plt.subplot(1, 2, 2)
    plt.scatter(unoc_pos[1], unoc_pos[0], s=2, c='green')
    plt.scatter(stats.mean(unoc_pos[1]), stats.mean(unoc_pos[0]), c='red', marker='x')
    plt.title('Unoccluded GPS position fixes', y=1.05)
    plt.xlabel('Longitude', x=0.4)
    plt.ylabel('Latitude')
    plt.legend(['Position', 'Mean'])

    # plot
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    # print statistics
    print('Occluded statistics:')
    print('XY mean:', stats.mean(oc_pos[1]), ',', stats.mean(oc_pos[0]))
    print('Z mean:', stats.mean(oc_pos[2]))
    print('X stddev:', stats.stdev(oc_pos[1]))
    print('Y stddev:', stats.stdev(oc_pos[0]))
    print('Z stddev:', stats.stdev(oc_pos[2]))

    print()
    print('Unoccluded statistics:')
    print('XY mean:', stats.mean(unoc_pos[1]), ',', stats.mean(unoc_pos[0]))
    print('Z mean:', stats.mean(unoc_pos[2]))
    print('X stddev:', stats.stdev(unoc_pos[1]))
    print('Y stddev:', stats.stdev(unoc_pos[0]))
    print('Z stddev:', stats.stdev(unoc_pos[2]))