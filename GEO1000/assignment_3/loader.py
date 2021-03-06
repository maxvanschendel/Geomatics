# GEO1000 - Assignment 3
# Authors:
# Studentnumbers:


def get_payload(raw_msg):
    """
    Returns tuple of payload and padding of a given raw AIS message

    For the raw AIS message: 

        !AIVDM,1,1,1,A,13an?n002APDdH0Mb85;8'sn06sd,0*76

    the payload is:

        13an?n002APDdH0Mb85;8'sn06sd

    the padding (the digit before the *) is:

        0

    Returns:
        tuple (payload:str, padding:int)
    """

    return raw_msg[14:-6], raw_msg[-5]


def read_payloads(filenm):
    """
    Reads the AIS messages (timestamp, payload and padding) from the file

    Arguments:
        :filenm: name of the file to be opened

    Uses:
        get_payload to get the payload and padding from each raw AIS message

    Returns:
        A list with tuples:
        [(timestamp:str, payload:str, padding:int), ...]
    """
    with open(filenm) as text_file:
        messages = text_file.readlines()

    out = []

    for i in messages:
        msg_split = i.split('\t')
        stamp = msg_split[0]
        pay, pad = get_payload(msg_split[1])[0], get_payload(msg_split[1])[1]
        out.append((stamp, pay, pad))

    return out


def _test():
    # use this function to test your implementation
    print(read_payloads('aislog.txt'))


if __name__ == "__main__":
    _test()
