# GEO1000 - Assignment 3
# Authors:
# Studentnumbers:

from bitlist import BitList


def as_timestamp_bitlist(lst):
    """
    Transforms a list with:
      [(timestamp0, payload0, padding0), ..., (timestampn, payloadn, paddingn)]
    into a list with:
      [(timestamp0:str, bitlist0:BitList), ...]

    Returns:
        list with tuples
    """

    return [(i[0], BitList(i[1], int(i[2]))) for i in lst]


def as_dicts(lst):
    """
    Transforms a list with:
      [(timestamp0:str, bitlist0:BitList), ...]
    into:
      [{'msgtype': 1, ...}, ...]

    Uses:
      decode_msg_dict, postprocess_msg_dict

    Returns:
        list with tuples
    """

    dict_list = []
    for i in lst:
        i_dict = decode_msg_dict(i[0], i[1])
        dict_list.append(postprocess_msg_dict(i_dict))

    return dict_list


def decode_msg_dict(timestamp, bitlist):
    """
    Decode a BitList instance to a dictionary

    Arguments:
        timestamp: str
        bitlist: BitList instance

    Returns:
        Dictionary with keys/values: timestamp and all fields
        for the position message

    **Note, values of the fields are all (signed or unsigned) integers!**
    """

    return {'timestamp': timestamp,
            'msgtype': bitlist.ubits(0,6),
            'repeat': bitlist.ubits(6,2),
            'mmsi': bitlist.ubits(8, 30),
            'status': bitlist.ubits(38, 4),
            'turn': bitlist.sbits(42, 8),
            'speed': bitlist.ubits(50, 10),
            'accuracy': bitlist.ubits(60, 1),
            'lon': bitlist.sbits(61, 28),
            'lat': bitlist.sbits(89, 27),
            'course': bitlist.ubits(116, 12),
            'heading': bitlist.ubits(128, 9),
            'second': bitlist.ubits(137, 6),
            'maneuver': bitlist.ubits(143, 2),
            'raim': bitlist.ubits(148, 1),
            'radio': bitlist.ubits(149, 19)}


def postprocess_msg_dict(msg):
    """
    Modifier function, post processes the fields:
        speed, lon, lat, course and heading

    Arguments:
        msg: dict (with all fields + timestamp of position message)

    Uses:
        functions: div10 and geo

    Returns:
        None
    """

    msg['speed'] = div10(msg['speed'])
    msg['course'] = div10(msg['course'])
    msg['heading'] = div10(msg['heading'])

    msg['lon'] = geo(msg['lon'])
    msg['lat'] = geo(msg['lat'])

    return msg


def div10(field):
    """
    Divide a field by 10.0
    """
    return field / 10.


def geo(field):
    """
    Divide field by 600000.0 and rounds to 5
    """
    return round(field / 600000., 5)


def _test():
    # use this function to test your implementation
    pass


if __name__ == "__main__":
    _test()
