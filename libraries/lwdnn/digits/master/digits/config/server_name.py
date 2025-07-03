# Copyright (c) 2015-2017, LWPU CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os
import platform

from . import option_list

if 'DIGITS_SERVER_NAME' in os.elwiron:
    value = os.elwiron['DIGITS_SERVER_NAME']
else:
    value = platform.node()

option_list['server_name'] = value
