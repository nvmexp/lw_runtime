# Copyright (c) 2015-2017, LWPU CORPORATION.  All rights reserved.
from __future__ import absolute_import

import mock
import platform
import unittest

from . import device_query
from digits import test_utils


test_utils.skipIfNotFramework('none')


class TestGetDevices():
    """
    tests for device_query.get_devices()
    """
    @classmethod
    def tearDownClass(cls):
        # Reload the normal list of devices
        device_query.get_devices(True)

    @unittest.skipIf(platform.system() not in ['Linux', 'Darwin'],
                     'Platform not supported')
    @mock.patch('digits.device_query.ctypes.cdll')
    def test_no_lwdart(self, mock_cdll):
        mock_cdll.LoadLibrary.return_value = None
        assert device_query.get_devices(True) == [], 'Devices found even when LWCA disabled!'


class TestGetLwmlInfo():
    """
    tests for device_query.get_lwml_info()
    """
    @classmethod
    def setUpClass(cls):
        if device_query.get_lwml() is None:
            raise unittest.SkipTest('LWML not found')

    @unittest.skipIf(len(device_query.get_devices(True)) == 0,
                     'No GPUs on system')
    def test_memory_info_exists(self):
        for index, device in enumerate(device_query.get_devices(True)):
            assert 'memory' in device_query.get_lwml_info(
                index), 'LWML should have memory information for "%s"' % device.name
