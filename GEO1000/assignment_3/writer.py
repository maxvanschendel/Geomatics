# GEO1000 - Assignment 3
# Authors:
# Studentnumbers:

def write_tsv(lst, filenm_out):
    """
    Writes Tab Separated values to a file with name filenm_out

    Arguments:

        lst: list of dictionaries with the message content
             [{'msgtype': 1, ...}, {...}, ...]
        filenm_out: string specifying name of the file to use for output

    """
    with open(filenm_out, 'a+') as f:
        f.write('timestamp\tmsgtype\trepeat\tmmsi\tstatus\tturn\tspeed\taccuracy\tlon\tlat\tcourse\theading\tsecond\tmaneuver\traim\tradio\n')
        for i in lst:
            for k in i.keys():
                f.write(str(i[k]) + '\t')

            f.write('\n')







def _test():
    # use this function to test your implementation
    pass


if __name__ == "__main__":
    _test()
