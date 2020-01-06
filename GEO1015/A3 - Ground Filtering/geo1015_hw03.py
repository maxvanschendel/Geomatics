#-- main.py
#-- GEO1015.2019--hw03
#-- Ravi Peters <r.y.peters@tudelft.nl>

#------------------------------------------------------------------------------
# DO NOT MODIFY THIS FILE!!!
#------------------------------------------------------------------------------

import json, sys
from my_code_hw03 import filter_ground

def main():
    jparams = json.load(open('params.json'))

    filter_ground(jparams)

if __name__ == "__main__":
    main()