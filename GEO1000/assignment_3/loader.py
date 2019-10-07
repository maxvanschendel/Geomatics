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
    pass


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
    pass


def _test():
    # use this function to test your implementation
    pass


if __name__ == "__main__":
    _test()
