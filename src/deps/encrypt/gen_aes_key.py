#!/usr/bin/elw python3

#
# LWIDIA_COPYRIGHT_BEGIN
#
# Copyright 2019 by LWPU Corporation. All rights
# reserved. All information contained herein is proprietary and confidential to
# LWPU Corporation.  Any use, reproduction, or disclosure without the written
# permission of LWPU Corporation is prohibited.
#
# LWIDIA_COPYRIGHT_END
#

import random
import sys

with open(sys.argv[2], "w") as f:
    num_blocks = 3
    array_name = sys.argv[1]
    print("extern const unsigned char {}[16 * {}];".format(array_name, num_blocks), file=f)
    print("const unsigned char %s[16 * %d] = {" % (array_name, num_blocks), file=f)

    for line_idx in range(num_blocks):
        vals = "    "
        vals += ", ".join(["0x{:02x}".format(random.randint(0, 255)) for jj in range(16)])
        if line_idx < num_blocks - 1:
            vals += ","
        print(vals, file=f)

    print("};", file=f)
