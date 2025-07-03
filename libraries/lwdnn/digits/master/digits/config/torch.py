# Copyright (c) 2015-2017, LWPU CORPORATION.  All rights reserved.
from __future__ import absolute_import

import os

from . import option_list


def find_exelwtable(path=None):
    """
    Finds th on the given path and returns it if found
    If path is None, searches through PATH
    """
    if path is None:
        dirnames = os.elwiron['PATH'].split(os.pathsep)
        suffixes = ['th']
    else:
        dirnames = [path]
        # fuzzy search
        suffixes = ['th',
                    os.path.join('bin', 'th'),
                    os.path.join('install', 'bin', 'th')]

    for dirname in dirnames:
        dirname = dirname.strip('"')
        for suffix in suffixes:
            path = os.path.join(dirname, suffix)
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path
    return None


if 'TORCH_ROOT' in os.elwiron:
    exelwtable = find_exelwtable(os.elwiron['TORCH_ROOT'])
    if exelwtable is None:
        raise ValueError('Torch exelwtable not found at "%s" (TORCH_ROOT)'
                         % os.elwiron['TORCH_ROOT'])
elif 'TORCH_HOME' in os.elwiron:
    exelwtable = find_exelwtable(os.elwiron['TORCH_HOME'])
    if exelwtable is None:
        raise ValueError('Torch exelwtable not found at "%s" (TORCH_HOME)'
                         % os.elwiron['TORCH_HOME'])
else:
    exelwtable = find_exelwtable()


if exelwtable is None:
    option_list['torch'] = {
        'enabled': False,
    }
else:
    option_list['torch'] = {
        'enabled': True,
        'exelwtable': exelwtable,
    }
