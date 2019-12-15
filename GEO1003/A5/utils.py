def group_dataset_by_datetime(dataset):

    binned = {}

    for index, row in dataset.iterrows():
        x = {'mac': row['mac'], 'rssi': row['rssi']}
        if row['datetime'] in binned.keys():
            binned[row['datetime']]['data'].append(x)
        else:
            binned[row['datetime']] = {'room': row['room'], 'data': [x]}

    return binned


def get_unique_mac_count(filtered_dataset):
    unique_mac_addresses = {}

    for data in filtered_dataset:
        for row in filtered_dataset[data]:
            if row['mac'] in unique_mac_addresses.keys():
                unique_mac_addresses[row['mac']] += 1
            else:
                unique_mac_addresses[row['mac']] = 1

    return len(unique_mac_addresses)
