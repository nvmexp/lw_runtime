# Copyright (c) 2016-2017, LWPU CORPORATION.  All rights reserved.
from __future__ import absolute_import

from digits.utils import subclass
from flask_wtf import Form


@subclass
class ConfigForm(Form):
    pass
