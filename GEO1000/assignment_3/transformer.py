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

    return [(i[0], BitList(i[1], i[2])) for i in lst]


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
    pass


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
    pass


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
    pass


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
