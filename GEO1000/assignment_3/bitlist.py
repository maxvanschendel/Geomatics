from array import array

__all__ = ['BitList']
__license__ = "BSD license"

# Based on the AIS parser in GPSD
_BITS_PER_BYTE = 8

class BitList(object):
    """
    Class that stores a list of bits (based on Python built-in array type)
    of an AIS message.

    Note, that a developer does not need to know how the bits are stored,
    to access them with the ubits and sbits methods. This way, the concept
    of *information hiding* is realised.

    See Section 17.11 of Think Python for more information.
    """

    def __init__(self, data, pad=0):
        """
        Copy characters from AIVDM-style six-bit character string and set
        padding (reduces number of bits to consider in the bit list).
        """
        self._bits = array('B')  # each entry in this array contains 8 bits
        self._bitlen = 0
        for ch in data:
            self._add_char(ch)
        self._add_padding(pad)

    def _add_char(self, char):
        """Add a character from an NMEA string to the list of bits"""
        # make sure the byte array has sufficient length
        # to store 6 additional bits
        current_length = len(self._bits) * 8
        needed_length = (self._bitlen + 6)
        if needed_length > current_length:
            self._bits.extend([0])
        # value of the char in the ASCII table, minus 48 and if still above 40
        # minus 8, leading to the integer value needed
        ch = ord(char) - 48
        if ch > 40:
            ch -= 8
        # ch contains the correct numeric value, for which we have 6 bits
        # that we want to copy over onto the array of bytes (where each entry
        # has 8 bits)
        for i in (5, 4, 3, 2, 1, 0):
            # if the i-th bit in the 6-bit ch value is set to 1
            # we copy this 1 over to the correct location in the byte array
            if (ch >> i) & 0x01:
                self._bits[self._bitlen // 8] |= (1 << (7 - self._bitlen % 8))
            # record that we have 1 more bit added in the byte array
            self._bitlen += 1

    def _add_padding(self, pad):
        """Set the correct padding"""
        # padding reduces the number of bits to consider in the payload
        self._bitlen -= pad

    def ubits(self, start, width):
        """
        Extract a (zero-origin) bitfield from the byte buffer as an unsigned int

        Arguments:
            :start: start bit
            :width: number of bits to consider

        Returns: zero or positive integer
        """
        fld = 0
        for i in range((start // _BITS_PER_BYTE),
                       (start + width + _BITS_PER_BYTE - 1) // _BITS_PER_BYTE):
            fld <<= _BITS_PER_BYTE  # shift left
            fld |= self._bits[i]     # bitwise or
        end = (start + width) % _BITS_PER_BYTE
        if end != 0:
            fld >>= (_BITS_PER_BYTE - end)  # shift right
        fld &= ~(-1 << width)  # make an integer which has 'width' bits set to 1
                               # preserve those bits that are needed in the
                               # final fld by performing a bitwise and
        return fld

    def sbits(self, start, width):
        """
        Extract a (zero-origin) bitfield from the buffer as a signed int

        Arguments:
            :start: start bit
            :width: number of bits to consider

        Returns: zero, positive or negative integer
        """
        # get the bits as unsigned integer
        fld = self.ubits(start, width)
        # if the highest bit of this unsigned integer is 1,
        # this actually is a negative number
        if fld & (1 << (width - 1)):
            # hence, we calculate the max value for this integer and
            # subtract fld from this max value and return it as negative int
            fld = -(2 ** width - fld)
        return fld

    def __len__(self):
        """How many bits are stored in this BitList."""
        return self._bitlen

    def __repr__(self):
        """Shows the stored data

        The resulting string has the number of bits and then
        the string of bits (string with 1's and 0's)"""
        format_byte = lambda _: format(_, 'b').zfill(8)  # anonymous function
        bits_str = "".join(map(format_byte, self._bits))[:self._bitlen]
        return str(self._bitlen) + ":" + bits_str


if __name__ == "__main__":
    # sample payload + padding
    payload, padding = ("13an?n002APDdH0Mb85;8'sn06sd", 0)

    # construct a list of bits based on the payload and padding
    bitlist = BitList(payload, padding)

    # print its contents
    print bitlist
    # get the 6 bits starting at position 0 as an unsigned integer
    print bitlist.ubits(0, 6)
