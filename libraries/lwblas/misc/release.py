#!/usr/bin/elw python3

import sys
import io

if __name__ == "__main__":
    """
    Parses types.h and removes all lines with {$lw-internal-release}
    """

    files = ['./include/lwtensor/types.h', './include/lwtensor.h']

    lwtensor_root = sys.argv[1]
    output_dir = sys.argv[2]
    for f in files:
        with io.open(output_dir + "/" + f.split('/')[-1], 'w', encoding='utf-8') as file_out, \
             io.open(lwtensor_root + "/" + f, 'r', encoding='utf-8') as file_in:
            for line in file_in:
                if("{$lw-internal-release}" not in line):
                    file_out.write(line)
