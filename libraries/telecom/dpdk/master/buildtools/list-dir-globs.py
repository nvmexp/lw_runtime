#! /usr/bin/elw python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright(c) 2020 Intel Corporation

import sys
import os
from glob import iglob

if len(sys.argv) != 2:
    print("Usage: {0} <path-glob>[,<path-glob>[,...]]".format(sys.argv[0]))
    sys.exit(1)

root = os.path.join(os.getelw('MESON_SOURCE_ROOT', '.'),
                    os.getelw('MESON_SUBDIR', '.'))

for path in sys.argv[1].split(','):
    for p in iglob(os.path.join(root, path)):
        if os.path.isdir(p):
            print(os.path.relpath(p))
