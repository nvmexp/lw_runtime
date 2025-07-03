#!/usr/bin/elw python
# -*- coding: utf-8 -*-

import os
import re
import sys
import uuid
import glob
import time
import ctypes
import random
import getpass
import itertools
import threading
import unittest
from ctypes import *
from sets import Set

import Utils
from teamcity import is_running_under_teamcity
from teamcity.unittestpy import TeamcityTestRunner
from Config import *
from Utils import *

LogFile = 'YAMLFXCTests.log'

@unittest.skip("disabling these test for now")
@for_all_test_methods(catch_all_errors)
@for_all_assert_methods(count_asserts)
class YAMLFXCTests(unittest.TestCase):
	def test_determenism(self):
		for i in range(100):
			run_detached_process([YAMLFXC32Path, '--in', "C:\Program Files\LWPU Corporation\Ansel\ShaderMod\BlacknWhite.yaml", '--out', '1.acef'], '.').wait()
			run_detached_process([YAMLFXC32Path, '--in', "C:\Program Files\LWPU Corporation\Ansel\ShaderMod\BlacknWhite.yaml", '--out', '2.acef'], '.').wait()
			self.assertTrue(compare_files('1.acef', '2.acef'))

			run_detached_process([YAMLFXC64Path, '--in', "C:\Program Files\LWPU Corporation\Ansel\ShaderMod\BlacknWhite.yaml", '--out', '3.acef'], '.').wait()
			run_detached_process([YAMLFXC64Path, '--in', "C:\Program Files\LWPU Corporation\Ansel\ShaderMod\BlacknWhite.yaml", '--out', '4.acef'], '.').wait()
			self.assertTrue(compare_files('3.acef', '4.acef'))

try:
	if __name__ == '__main__':
		if is_running_under_teamcity():
			runner = TeamcityTestRunner()
			unittest.main(testRunner=runner)
		else:
			with open(LogFile, "w") as f:
				runner = unittest.TextTestRunner(f)
				unittest.main(testRunner=runner, exit=False)
				print ('Assertions statistics: %d passed, %d failed (%d total)' % (Utils.AssertionPassCounter, Utils.AssertionFailCounter, Utils.AssertionCounter))
finally:
	pass
