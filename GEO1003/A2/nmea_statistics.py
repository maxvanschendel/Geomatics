import matplotlib.pyplot as plt
import statistics as stats
import mplleaflet
import pynmea2


def combine_data(files, newfile_name):
    with open(newfile_name, 'a') as nf:
        for i in files:
            with open(i) as source_file:
                nf.write(source_file.read())


def extract_position_fixes(file):
    with open(file) as open_file:
        file_content = open_file.readlines()
        data = [pynmea2.parse(msg) for msg in file_content]

        rmc = list(filter(lambda x: x.sentence_type == 'RMC' and x.status == 'A', data))
        gga = list(filter(lambda x: x.sentence_type == 'GGA' and x.gps_qual != 0, data))
        fixes = rmc + gga

        lat = [msg.latitude for msg in fixes]
        lon = [msg.longitude for msg in fixes]
        alt = [msg.altitude for msg in gga]

    return lat, lon, alt


def pos_statistics(x_pop, y_pop, z_pop):
    print('XY mean:', stats.mean(x_pop), ',', stats.mean(y_pop))
    print('XY median:', stats.median(x_pop), ',', stats.median(y_pop))
    print('Z mean:', stats.mean(z_pop), 'm')
    print('X stddev:', stats.stdev(x_pop))
    print('Y stddev:', stats.stdev(y_pop))
    print('Z stddev:', stats.stdev(z_pop), 'm')


if __name__ == '__main__':
    # unoccluded and occluded data files
    unoc = './data/unoccluded.txt'
    oc = './data/occluded.txt'

    # combine seperate files into one dataset
    #combine_data(['./data/a.txt', './data/b.txt', './data/c.txt'], unoc)
    #combine_data(['./data/a2.txt', './data/b2.txt', './data/c2.txt'], oc)

    # extract GGA latitude and longitude data
    unoc_pos = extract_position_fixes(unoc)
    oc_pos = extract_position_fixes(oc)

    #create scatterplot of data
    ax1 = plt.subplot(1, 2, 1)
    plt.scatter(oc_pos[1], oc_pos[0], s=2, c='green')

    plt.scatter(stats.mean(oc_pos[1]), stats.mean(oc_pos[0]), c='red', marker='x')
    plt.scatter(stats.median(oc_pos[1]), stats.median(oc_pos[0]), c='blue', marker='x')
    plt.title('Occluded GPS position fixes', y=1.05)
    plt.xlabel('Longitude', x=0.4)
    plt.ylabel('Latitude')
    plt.legend(['Pos. Fix', 'Mean', 'Median'])

    # unoccluded data
    ax2 = plt.subplot(1, 2, 2)
    plt.scatter(unoc_pos[1], unoc_pos[0], s=2, c='green')
    plt.scatter(stats.mean(unoc_pos[1]), stats.mean(unoc_pos[0]), c='red', marker='x')
    plt.scatter(stats.median(unoc_pos[1]), stats.median(unoc_pos[0]), c='blue', marker='x')
    plt.title('Unoccluded GPS position fixes', y=1.05)
    plt.xlabel('Longitude', x=0.4)
    plt.ylabel('Latitude')

    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    plt.show()

    # leafly plot
    fig = plt.figure()

    plt.plot(unoc_pos[1], unoc_pos[0])
    plt.plot(unoc_pos[1], unoc_pos[0], 'r.')

    plt.plot(oc_pos[1], oc_pos[0])
    plt.plot(oc_pos[1], oc_pos[0], 'r.')

    mplleaflet.show()

    # print statistics
    print('Occluded statistics:')
    pos_statistics(oc_pos[1], oc_pos[0], oc_pos[2])

    print()
    print('Unoccluded statistics:')
    pos_statistics(unoc_pos[1], unoc_pos[0], unoc_pos[2])
